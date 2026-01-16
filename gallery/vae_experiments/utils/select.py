import numpy as np
import torch

def xgb_select_indices(xgb_all_preds, train_indices, test_indices, topk_size, eps_greedy_size, rng):
    """
    랜덤으로 2개, xgb 모델로 상위 62개 선택
    """
    # 남은 인덱스 중에서 무작위로 random_select_size개 선택

    remaining_indices = set(test_indices)

    if topk_size + eps_greedy_size > test_indices.shape[0]:
        remaining_indices.update(train_indices.tolist())
        train_indices = np.array(list(remaining_indices), dtype=np.int64)
        return train_indices, np.array([], dtype=np.int64)


    top_indices, remaining_indices = select_topk_cost(xgb_all_preds, remaining_indices, topk_size)
    random_indices, remaining_indices = random_select_indices(remaining_indices, eps_greedy_size, rng=rng)
    test_indices = np.array(list(remaining_indices), dtype=np.int64)

    selected_indices = np.concatenate([top_indices, random_indices])

    train_indices = np.concatenate([train_indices, selected_indices])

    return train_indices, test_indices



def random_select_indices(remaining_indices, select_size, rng=np.random.default_rng(42)):
    if select_size == 0:
        return np.array([], dtype=np.int64), remaining_indices
    
    random_indices = rng.choice(list(remaining_indices), size=select_size, replace=False)

    remaining_indices = util_update_remaining_indices(remaining_indices, random_indices)

    return random_indices, remaining_indices



def util_update_remaining_indices(remaining_indices, selected_indices):
    """
    남은 인덱스 집합 업데이트
    util_update_remaining_indices에서 selected_indices 제거
    """
    selected_indices = set(selected_indices)
    remaining_indices.difference_update(selected_indices)

    return remaining_indices



def util_select_topk(predictions, remaining_indices, num_select):
    """
    예측값 기반 다음 측정할 샘플 선택
    
    Args:
        predictions: 전체 예측값 리스트 ([N, ] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 numpy 배열
        remaining_indices: 업데이트된 남은 인덱스 집합 (set)
    """
    
    prediction = np.asarray(predictions)  # [N]

    remaining_np = np.array(list(remaining_indices), dtype=np.int64)
    remaining_pred = prediction[remaining_np]

    k = min(num_select, len(remaining_np))

    topk_local = np.argsort(remaining_pred)[-k:]
    selected_indices = remaining_np[topk_local]

    # remaining 업데이트
    remaining_indices.difference_update(selected_indices.tolist())

    return selected_indices, remaining_indices






def select_topk_cost(cost_pred, remaining_indices, num_select):
    """
    예측된 cost 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices

    if isinstance(cost_pred, torch.Tensor):
        cost_pred = cost_pred.detach().cpu().numpy()  # [N]

    topk_cost_indices, remaining_indices = util_select_topk(cost_pred, remaining_indices, num_select)
    

    return topk_cost_indices, remaining_indices


def select_topk_z_grad(z, cost_pred, remaining_indices, num_select):
    """
    z에 대한 cost gradient 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_tensor: 전체 input numpy 배열 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
    
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices

    candidate_indices = np.array(list(remaining_indices), dtype=np.int64)

    # z-gradient 계산
    z_grad = torch.autograd.grad(
        outputs=cost_pred.sum(),
        inputs=z,
        retain_graph=False,
        create_graph=False
    )[0]  # [N, latent_dim]

    z_grad_norm = torch.norm(z_grad, dim=1).detach().cpu().numpy()  # [N]

    # 후보 중 grad-norm top-k
    candidate_grad = z_grad_norm[candidate_indices]
    k = min(num_select, len(candidate_indices))

    topk_local = np.argsort(candidate_grad)[-k:]
    selected_indices = candidate_indices[topk_local]

    # remaining 업데이트
    remaining_indices = set(remaining_indices)
    remaining_indices.difference_update(selected_indices.tolist())

    return selected_indices, remaining_indices


def select_topk_uncertainty(model, input_tensor, remaining_indices, num_select, T_mc=10):
    """
    MC Dropout 기반 불확실성 추정으로 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """
    if num_select == 0:
        return np.array([], dtype=np.int64), remaining_indices


    was_training = model.training
    model.train()

    with torch.no_grad():
        _, mc_var = model.mc_predict(input_tensor, T=T_mc)

    if not was_training:
        model.eval()  # 원복

    var_np = mc_var.detach().cpu().numpy()  # [N]

    topk_uncertainty_indices, remaining_indices = util_select_topk(var_np, remaining_indices, num_select)

    return topk_uncertainty_indices, remaining_indices


def select_topk_latent_diversity(z, candidate_indices, used_indices, select_n_div, chunk_size=1024, eps=1e-12):
    """
    먼저 candidates 320개를 뽑았다고 치자.
    이후 앞에서 topk_cost, topk_z_grad로 40개 정도를 뽑았다고 치자.
    latent diversity는 40개 + used_indices로부터 가장 멀리 떨어진 24개를 280개에서 뽑는다.

    z를 L2 정규화한 뒤, k-center greedy(farthest-first)로 diversity 선택.
    초기 센터는 used_indices (이미 측정된 점들).
    매 스텝마다 "센터 집합까지의 최소거리"가 최대인 candidate를 하나씩 추가.
    
    Args:
        z: torch.Tensor [N, latent_dim]
        candidate_indices: set(int)
        used_indices: set(int)
        select_n_div: int
        chunk_size: int
    Returns:
        diverse_indices: np.ndarray (int64)
        candidate_indices: set (선택된 인덱스 제거된 상태)
    """
    if select_n_div == 0 or len(candidate_indices) == 0:
        return np.array([], dtype=np.int64), candidate_indices


    device = z.device

    # 1) L2 normalize z  (각 벡터를 단위벡터로)
    with torch.no_grad():
        z_norm = z / (z.norm(dim=1, keepdim=True) + eps)

    cand = np.array(list(candidate_indices), dtype=np.int64)
    k = min(select_n_div, len(cand))

    cand_t = torch.from_numpy(cand).to(device=device)
    z_cand = z_norm[cand_t]  # [M, D], M=len(cand)

    # 초기 센터: used_indices (비어있을 수도 있음)
    used = np.array(list(used_indices), dtype=np.int64)
    selected = []

    # 2) 각 candidate의 "현재 센터 집합까지 최소거리" 벡터 init
    #    used가 비어있으면 +inf로 시작해서 임의 첫 점을 뽑게(가장 큰 값) 만들기
    if len(used) > 0:
        used_t = torch.from_numpy(used).to(device=device)
        z_used = z_norm[used_t]  # [U, D]

        # min_dists[j] = min_{u in used} ||z_cand[j] - z_used[u]||
        min_dists = torch.empty(len(cand), device=device, dtype=torch.float32)

        with torch.no_grad():
            for s in range(0, len(cand), chunk_size):
                e = min(s + chunk_size, len(cand))
                d = torch.cdist(z_cand[s:e], z_used, p=2)  # [B, U]
                min_dists[s:e] = d.min(dim=1).values
    else:
        # 센터가 없으면 모두 동일하게 시작 → 첫 선택은 아래 argmax가 0번째로 갈 수 있음
        # 다양성 목적이면 랜덤/최대 norm 등으로 첫 점을 정할 수도 있지만,
        # 여기서는 "가장 큰 min_dists"를 위해 +inf로 둔다.
        min_dists = torch.full((len(cand),), float("inf"), device=device, dtype=torch.float32)

    # 3) k-center greedy 반복
    #    매번 argmax(min_dists) 하나 선택 -> 그 점을 센터에 추가 -> min_dists 갱신
    with torch.no_grad():
        for _ in range(k):
            j = torch.argmax(min_dists).item()     # cand 내부 위치
            sel_idx = cand[j]                      # 원본 인덱스
            selected.append(sel_idx)

            # 선택된 점을 "센터"로 추가: 모든 candidate에 대해 dist_to_new_center 계산 후 min 갱신
            new_center = z_cand[j:j+1]  # [1, D]

            # 방금 뽑은 점은 다시 뽑히지 않게 min_dists를 -inf로
            min_dists[j] = -float("inf")

            # 나머지 후보들의 min 거리 업데이트
            for s in range(0, len(cand), chunk_size):
                e = min(s + chunk_size, len(cand))
                d_new = torch.cdist(z_cand[s:e], new_center, p=2).squeeze(1)  # [B]
                min_dists[s:e] = torch.minimum(min_dists[s:e], d_new)

    diverse_indices = np.array(selected, dtype=np.int64)

    candidate_indices = set(candidate_indices)
    candidate_indices.difference_update(diverse_indices.tolist())

    return diverse_indices, candidate_indices






def select_programs(model, input_data_scaled, used_indices, remaining_indices, num_select=64, T_mc=10, uncertainty_topk=128,
                    w_cost=0.5, w_unc=0.3, w_div=0.2, grad_num=2, rand_num=0, rng=np.random.default_rng(42), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), topk_factor=5):
    """
    Active learning 기반 다음 측정할 샘플 선택
    
    Args:
        model: VAECostPredictor 모델
        input_data_scaled: 전체 input 리스트 ([N, input_dim] 형태)
        used_indices: 이미 측정된 인덱스 집합(set)
        remaining_indices: 아직 측정되지 않은 인덱스 집합 (set)
        num_select: 선택할 샘플 수
        T_mc: MC Dropout 샘플 수
        w_cost: 예측값이 큰 샘플 가중치
        w_unc: epistemic 불확실성이 높은 샘플 가중치
        w_div: latent 다양성이 높은 샘플 가중치
        grad_num: z에 대한 cost의 gradient가 큰 샘플 수
        rand_num: 무작위로 선택할 샘플 수
    
    Returns:
        selected_indices: 선택된 샘플의 인덱스 리스트
    """

    # 합쳐서 64개 선택
    total = num_select
    budget = total - grad_num - rand_num

    # 랜덤 선택만 할 경우
    if num_select == 0 and rand_num > 0:
        rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num, rng=rng)
        return rand_indices, remaining_indices
    

    select_n_cost = int(budget * w_cost)
    select_n_unc  = int(budget * w_unc)
    select_n_div  = int(budget * w_div)
    select_n_grad = grad_num
    s = select_n_cost + select_n_unc + select_n_div
    if s < budget:
        select_n_cost += budget - s

    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32, device=device)
    

    model.eval()
    with torch.no_grad():
        z, _ = model.encode(input_tensor)
    z = z.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    cost_pred = model.predict_cost(z)
    cost_pred = cost_pred.view(-1)
    cost_np = cost_pred.detach().cpu().numpy()

    remaining_np = np.array(list(remaining_indices), dtype=np.int64)
    remaining_cost = cost_np[remaining_np]

    k_pref = min(len(remaining_np), total * topk_factor)
    top_local = np.argsort(remaining_cost)[-k_pref:]
    candidate_indices = set(remaining_np[top_local].tolist())  # 작업용 remaining

    # print(f"Candidate pool size: {len(candidate_indices)}")


    # 중복 방지용
    currently_used = set()
    topk_cost_indices, candidate_indices = select_topk_cost(cost_pred, candidate_indices, select_n_cost)
    currently_used.update(topk_cost_indices.tolist())
    z_grad_indices, candidate_indices = select_topk_z_grad(z, cost_pred, candidate_indices, select_n_grad)
    currently_used.update(z_grad_indices.tolist())

    # if len(used_indices) / len(input_data_scaled) >= 0.1:
    if len(used_indices) >= uncertainty_topk:
        uncertainty_indices, candidate_indices = select_topk_uncertainty(model, input_tensor, candidate_indices, select_n_unc, T_mc=T_mc)
    else:
        pool_for_uncertainty = set(remaining_indices)
        pool_for_uncertainty.difference_update(currently_used)
        uncertainty_indices, _ = select_topk_uncertainty(model, input_tensor, pool_for_uncertainty, select_n_unc, T_mc=T_mc)
        candidate_indices.difference_update(uncertainty_indices.tolist())


    currently_used.update(uncertainty_indices.tolist())
    used_local = set(used_indices)
    used_local.update(currently_used)

    diverse_indices, _ = select_topk_latent_diversity(z, candidate_indices, used_local, select_n_div)
    currently_used.update(diverse_indices.tolist())


    remaining_indices.difference_update(currently_used)


    rand_indices, remaining_indices = random_select_indices(remaining_indices, rand_num, rng=rng)
    currently_used.update(rand_indices.tolist())

    

    all_selected_indices = np.array(sorted(currently_used), dtype=np.int64)



    return all_selected_indices, remaining_indices




def select_init_latent_diversity(model, input_data_scaled, remaining_indices, select_num):
    """
    맨 처음에 latent diversity 기반으로 초기 샘플 선택할 때 사용
    전체 데이터에서 latent diversity가 높은 샘플 select_n_div개 선택

    z를 L2 정규화한 뒤, k-center greedy(farthest-first)로 diversity 선택.
    초기 센터는 used_indices (이미 측정된 점들).
    매 스텝마다 "센터 집합까지의 최소거리"가 최대인 candidate를 하나씩 추가.
    
    Args:
        z: torch.Tensor [N, latent_dim]
        candidate_indices: set(int)
        used_indices: set(int)
        select_n_div: int
        chunk_size: int
    Returns:
        diverse_indices: np.ndarray (int64)
        candidate_indices: set (선택된 인덱스 제거된 상태)
    """

    model.eval()

    input_tensor = torch.tensor(
        input_data_scaled,
        dtype=torch.float32,
        device=model.device
    )

    with torch.no_grad():
        z = model.encode(input_tensor, use_mean=True)  # (N, D)

    z = z.detach()
    N = z.size(0)
    select_num = min(select_num, N)

    selected = []
    # 1) 첫 점은 랜덤
    first = torch.randint(0, N, (1,), device=z.device).item()
    selected.append(first)

    # 2) 나머지는 farthest-point greedy
    dist = torch.cdist(z, z[[first]])[:, 0]  # (N,)

    for _ in range(1, select_num):
        idx = torch.argmax(dist).item()
        selected.append(idx)

        new_dist = torch.cdist(z, z[[idx]])[:, 0]
        dist = torch.minimum(dist, new_dist)

    remaining_indices.difference_update(diverse_indices)
    diverse_indices = np.array(selected, dtype=np.int64)
   
    return diverse_indices, remaining_indices