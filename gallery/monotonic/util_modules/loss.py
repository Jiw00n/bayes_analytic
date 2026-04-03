import torch
import torch.nn.functional as F



def reconstruction_loss(x_recon, x):
    """
    VAE 재구성 손실 (MSE)
    """
    return F.mse_loss(x_recon, x, reduction="mean")


def kld_loss(mean, logvar):
    """
    KL Divergence: q(z|x) || N(0, I)
    """
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return kld




def reg_loss(cost_pred, cost_true, loss_type='mse'):
    """
    기본 회귀 손실 (MSE 또는 MAE)
    """
    if cost_pred.shape != cost_true.shape:
        if cost_true.ndim == 2 and cost_true.size(1) == 1:
            cost_true = cost_true.view(-1)
        elif cost_pred.ndim == 2 and cost_pred.size(1) == 1:
            cost_pred = cost_pred.view(-1)

    if loss_type == 'mse':
        return F.mse_loss(cost_pred, cost_true)
    else:  # mae
        return F.l1_loss(cost_pred, cost_true)


def pair_loss(cost_pred, cost_true, margin=0.1):
    """
    Pairwise ranking loss: 실제 cost 순서를 예측이 유지하도록.
    cost_true[i] < cost_true[j] 이면 cost_pred[i] < cost_pred[j] + margin
    """
    batch_size = cost_pred.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=cost_pred.device)
    
    # 모든 쌍에 대해 ranking loss 계산
    idx = torch.arange(batch_size, device=cost_pred.device)
    i_idx, j_idx = torch.meshgrid(idx, idx, indexing='ij')
    mask = i_idx < j_idx  # upper triangular only
    
    pred_i = cost_pred[i_idx[mask]]
    pred_j = cost_pred[j_idx[mask]]
    true_i = cost_true[i_idx[mask]]
    true_j = cost_true[j_idx[mask]]
    
    # label: 1 if true_i < true_j, -1 otherwise
    labels = torch.sign(true_j - true_i).float()
    
    # Margin ranking loss
    loss = F.margin_ranking_loss(pred_j.view(-1), pred_i.view(-1), labels.view(-1), margin=margin)
    return loss





def smooth_loss(model, z, noise_std=0.1):
    """
    Smoothness loss: z에 작은 노이즈를 더했을 때 예측이 크게 변하지 않도록.
    """
    model.eval()
    with torch.no_grad():
        z_noisy = z + noise_std * torch.randn_like(z)
    
    cost_original = model.predict_cost(z)
    cost_noisy = model.predict_cost(z_noisy)
    
    smooth_loss = F.mse_loss(cost_original, cost_noisy)
    return smooth_loss



import torch

def lipschitz_pairwise_loss(y: torch.Tensor,
                            z: torch.Tensor,
                            L: float,
                            eps: float = 1e-12,
                            reduction: str = "mean",
                            exclude_self: bool = True) -> torch.Tensor:
    """
    Compute: sum_{i,j} max(0, |y_i - y_j| / ||z_i - z_j||_2 - L)

    Args:
        y: (N,) or (N,1) or (N,*)  -- if (N,*) it's treated as vector output and uses L2 for |y_i-y_j|
        z: (N, Dz)
        L: Lipschitz constant bound (float)
        eps: small constant to avoid division by zero
        reduction: "sum" | "mean" | "none"
        exclude_self: if True, ignore i==j terms

    Returns:
        scalar loss if reduction != "none", else (N, N) matrix of per-pair penalties
    """
    if y.dim() == 1:
        y_ = y[:, None]  # (N,1)
        y_is_scalar = True
    else:
        y_ = y
        y_is_scalar = False

    # Pairwise differences
    # z_diff: (N,N,Dz)
    z_diff = z[:, None, :] - z[None, :, :]
    z_dist = torch.linalg.norm(z_diff, dim=-1)  # (N,N)

    # y difference:
    # - scalar y: abs
    # - vector y: L2 norm
    y_diff = y_[:, None, ...] - y_[None, :, ...]
    if y_is_scalar and y_.shape[1] == 1:
        y_dist = y_diff.abs().squeeze(-1)  # (N,N)
    else:
        # Treat as vector output; use L2 norm across last dims
        y_dist = torch.linalg.norm(y_diff.reshape(y_diff.shape[0], y_diff.shape[1], -1), dim=-1)  # (N,N)

    # Ratio with safe denominator
    ratio = y_dist / (z_dist.clamp_min(eps))

    penalty = torch.relu(ratio - L)  # (N,N)

    if exclude_self:
        # remove diagonal
        penalty = penalty - torch.diag(torch.diagonal(penalty))

    if reduction == "none":
        return penalty
    elif reduction == "sum":
        return penalty.sum()
    elif reduction == "mean":
        if exclude_self:
            n = penalty.shape[0]
            denom = n * (n - 1)
            return penalty.sum() / max(denom, 1)
        return penalty.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")



def infonce_loss(
    z: torch.Tensor,      # (B, D)
    c: torch.Tensor,      # (B,)
    tau: float = 0.1,     # similarity temperature (필수 하이퍼 1개)
    tau_c: float = None,  # cost temperature (None이면 배치에서 자동 추정)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cost-weighted InfoNCE (최소 버전)
    - 입력: 임베딩 z, 실제 cost c
    - 하이퍼: tau(유사도 temperature)만 사실상 튜닝하면 됨
    - tau_c는 None이면 배치 내 |c_i-c_j|의 median으로 자동 설정

    L_i = -log( sum_{j!=i} w_ij * exp(sim_ij/tau) / sum_{k!=i} exp(sim_ik/tau) )
    w_ij = exp(-|c_i-c_j|/tau_c)
    """
    # print(z.shape, c.shape)
    c = c.view(-1)
    assert z.dim() == 2 and c.dim() == 1
    B = z.size(0)
    if B < 2:
        return z.new_tensor(0.0)

    # cosine similarity
    z = F.normalize(z, p=2, dim=1)
    sim = z @ z.t()  # (B, B)

    # exclude diagonal
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    mask = ~eye

    # cost diff
    dc = (c.view(-1, 1) - c.view(1, -1)).abs()  # (B, B)

    # auto tau_c: median of valid pairwise diffs
    if tau_c is None:
        tau_c = dc[mask].median().clamp_min(eps).item()
    else:
        tau_c = float(tau_c)
        if tau_c <= 0:
            raise ValueError("tau_c must be > 0")

    # weights
    w = torch.exp(-dc / max(tau_c, eps)) * mask.float()  # (B, B)

    # logits
    logits = sim / tau

    # log denominator: log sum_{k!=i} exp(logits_ik)
    neg_inf = torch.finfo(logits.dtype).min
    logits_den = logits.masked_fill(~mask, neg_inf)
    den = torch.logsumexp(logits_den, dim=1)  # (B,)

    # log numerator: log sum_{j!=i} w_ij * exp(logits_ij)
    # = logsumexp(logits + log(w))
    logits_num = (logits + torch.log(w.clamp_min(eps))).masked_fill(~mask, neg_inf)
    num = torch.logsumexp(logits_num, dim=1)  # (B,)

    loss = -(num - den)  # (B,)

    # safety: in case of NaNs
    loss = loss[torch.isfinite(loss)]
    return loss.mean() if loss.numel() > 0 else z.new_tensor(0.0)




# 부호 반대인데 성능이 더 잘 나왔음
def neg_ordered_infonce_loss(
    z: torch.Tensor,
    cost: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized minimal Ordered InfoNCE loss.

    positives(i): all j with cost_j < cost_i
    negatives(i): all j with cost_j > cost_i
    negatives get automatic margin: sim(i,j) - (cost_j - cost_i)/std(cost)

    Only tuning knob: temperature
    """
    assert z.dim() == 2
    assert cost.dim() == 1 and cost.shape[0] == z.shape[0]
    B = z.shape[0]
    device = z.device

    # Normalize embeddings -> cosine sim
    z = F.normalize(z, p=2, dim=1, eps=eps)
    sim = (z @ z.t()) / max(temperature, eps)  # (B,B)

    # Pairwise masks by cost order
    c_i = cost[:, None]        # (B,1)
    c_j = cost[None, :]        # (1,B)
    same = torch.eye(B, device=device, dtype=torch.bool)

    pos_mask = (c_j < c_i) & (~same)   # (B,B)
    neg_mask = (c_j > c_i) & (~same)   # (B,B)

    # Automatic margin for negatives
    scale = cost.std(unbiased=False).clamp_min(eps)
    delta = (c_j - c_i).clamp_min(0.0) / scale  # (B,B)
    neg_sim = sim - delta

    # We need:
    # num_i = logsumexp(sim[i, pos])
    # den_i = logsumexp([sim[i,pos], neg_sim[i,neg]])
    # Implement with masking by setting invalid entries to -inf
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=sim.dtype)

    pos_logits = sim.masked_fill(~pos_mask, neg_inf)          # (B,B)
    neg_logits = neg_sim.masked_fill(~neg_mask, neg_inf)      # (B,B)

    num = torch.logsumexp(pos_logits, dim=1)                  # (B,)
    den = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)  # (B,)

    # anchors that have at least one pos and one neg in batch
    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg

    # If no valid anchors (e.g., tiny batch), return 0 with grad
    if not valid.any():
        return z.sum() * 0.0

    loss = -(num - den)
    return loss[valid].mean()


# 이게 진짜인데 성능이 덜 나왔음
def real_ordered_infonce_loss(
    z: torch.Tensor,
    cost: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized minimal Ordered InfoNCE loss.

    positives(i): all j with cost_j > cost_i
    negatives(i): all j with cost_j < cost_i
    negatives get automatic margin: sim(i,j) - (cost_j - cost_i)/std(cost)

    Only tuning knob: temperature
    """
    assert z.dim() == 2
    assert cost.dim() == 1 and cost.shape[0] == z.shape[0]
    B = z.shape[0]
    device = z.device

    # Normalize embeddings -> cosine sim
    z = F.normalize(z, p=2, dim=1, eps=eps)
    sim = (z @ z.t()) / max(temperature, eps)  # (B,B)

    # Pairwise masks by cost order
    c_i = cost[:, None]        # (B,1)
    c_j = cost[None, :]        # (1,B)
    same = torch.eye(B, device=device, dtype=torch.bool)

    pos_mask = (c_j > c_i) & (~same)   # (B,B)
    neg_mask = (c_j < c_i) & (~same)   # (B,B)

    # Automatic margin for negatives
    scale = cost.std(unbiased=False).clamp_min(eps)
    delta = (c_i - c_j).clamp_min(0.0) / scale  # (B,B)
    neg_sim = sim - delta

    # We need:
    # num_i = logsumexp(sim[i, pos])
    # den_i = logsumexp([sim[i,pos], neg_sim[i,neg]])
    # Implement with masking by setting invalid entries to -inf
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=sim.dtype)

    pos_logits = sim.masked_fill(~pos_mask, neg_inf)          # (B,B)
    neg_logits = neg_sim.masked_fill(~neg_mask, neg_inf)      # (B,B)

    num = torch.logsumexp(pos_logits, dim=1)                  # (B,)
    den = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)  # (B,)

    # anchors that have at least one pos and one neg in batch
    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg

    # If no valid anchors (e.g., tiny batch), return 0 with grad
    if not valid.any():
        return z.sum() * 0.0

    loss = -(num - den)
    return loss[valid].mean()







def feature_loss(use_feature, feature_pred, feature_true, coef=0.1):
    """
    Feature 예측 손실 (MSE)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_feature:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(feature_pred, feature_true) * coef





# PCA 기반 방향성 loss
def pca_direction_loss(z: torch.Tensor, y: torch.Tensor, good_frac: float = 0.2, eps: float = 1e-12) -> torch.Tensor:
    """
    z: (N, Dz)
    y: (N,)  true cost (작을수록 good)
    good_frac: good subset 비율
    """
    
    @torch.no_grad()
    def pc1_direction(z_good: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # z_good: (Ng, Dz)
        zc = z_good - z_good.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(zc, full_matrices=False)
        d = Vh[0]
        d = d / (torch.norm(d) + eps)
        return d  # (Dz,)
    
    N = y.numel()
    k = max(4, int(N * good_frac))
    # cost 큰 것들 선택
    good_idx = torch.topk(y, k=k, largest=True).indices
    z_good = z[good_idx]
    y_good = y[good_idx]

    # PC1 방향 (SVD는 no_grad로 뽑아서 안정적으로)
    d = pc1_direction(z_good).detach()  # backprop through SVD 막기

    # projection
    p = (z_good @ d)  # (k,)

    # 부호 자동 선택:
    # p가 커질수록 cost가 "작아지게"(나빠지게) 정렬시키는 걸 기본으로 두자.
    # 만약 반대 상관이면 d를 뒤집어.
    p_center = p - p.mean()
    y_center = y_good - y_good.mean()
    corr = (p_center * y_center).sum()  # 스케일 상관없이 부호만 사용
    if corr < 0:
        d = -d
        p = -p
        p_center = -p_center  # y_center는 그대로

    # 이제 p와 y_good를 "로컬에서만" 정렬시키는 loss
    # 스케일/분산 차이 때문에 z-score로 맞추는 게 안전
    p_norm = (p - p.mean()) / (p.std() + eps)
    y_norm = (y_good - y_good.mean()) / (y_good.std() + eps)

    # L2로 맞추기(가장 단순, 안정적)
    return F.mse_loss(p_norm, y_norm)
