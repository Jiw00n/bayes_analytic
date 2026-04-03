
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import random

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def pairwise_dist_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((b - a.unsqueeze(0)) ** 2).sum(dim=-1)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a_n = a / (a.norm() + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return torch.matmul(b_n, a_n)

def retrieve_topk(
    z_query: torch.Tensor,
    Z: torch.Tensor,
    k: int = 1,
    metric: str = "l2",
    exclude: Optional[set] = None,
):
    if exclude is None:
        exclude = set()

    if metric == "l2":
        d = pairwise_dist_l2(z_query, Z)  # [N]
        if len(exclude) > 0:
            mask = torch.zeros_like(d, dtype=torch.bool)
            ex = torch.tensor(list(exclude), device=d.device, dtype=torch.long)
            mask[ex] = True
            d = d.masked_fill(mask, float("inf"))
        vals, idxs = torch.topk(d, k=min(k, Z.shape[0]), largest=False)
        return idxs, vals

    if metric == "cos":
        s = cosine_sim(z_query, Z)  # [N]
        if len(exclude) > 0:
            mask = torch.zeros_like(s, dtype=torch.bool)
            ex = torch.tensor(list(exclude), device=s.device, dtype=torch.long)
            mask[ex] = True
            s = s.masked_fill(mask, float("-inf"))
        vals, idxs = torch.topk(s, k=min(k, Z.shape[0]), largest=True)
        return idxs, vals

    raise ValueError(f"Unknown metric: {metric}")

@dataclass
class OneStepCfg:
    # cost는 "클수록 좋음"(negative log) 가정
    good_q: float = 0.10
    num_seeds: int = 200

    M: int = 32
    eps: float = 0.2

    retrieve_metric: str = "l2"  # "l2" or "cos"
    retrieve_k: int = 1

    baseline_query_mode: str = "sample"  # "sample" or "gauss"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def _sample_good_seeds_from_train(cost_train: np.ndarray, good_q: float, num_seeds: int, seed: int) -> np.ndarray:
    set_seed(seed)
    thr = np.quantile(cost_train, 1.0 - good_q)  # 상위 good_q
    good_idx = np.where(cost_train >= thr)[0]
    if len(good_idx) == 0:
        raise RuntimeError("train에서 good seed가 0개임. good_q 조절 필요.")
    if len(good_idx) < num_seeds:
        # print(f"[warn] train good seed가 {len(good_idx)}개뿐이라 전부 사용함.")
        return good_idx
    return np.random.choice(good_idx, size=num_seeds, replace=False)

def _perturb_l2_ball(z: torch.Tensor, eps: float) -> torch.Tensor:
    d = z.shape[0]
    u = torch.randn(d, device=z.device)
    u = u / (u.norm() + 1e-12)
    return z + eps * u

def one_step_neighborhood_test_train_seed_test_pool(
    Z_train: torch.Tensor,
    cost_train: np.ndarray,
    Z_test: torch.Tensor,
    cost_test: np.ndarray,
    cfg: OneStepCfg,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    seed는 train에서 선택, 후보검색/평가는 test에서만 수행.
    반환값 의미:
      - seed_cost는 'train seed의 cost'
      - neigh_* / rand_*는 'test에서 선택된 프로그램들의 cost'
    """
    assert Z_train.shape[0] == cost_train.shape[0]
    assert Z_test.shape[0] == cost_test.shape[0]

    set_seed(seed)
    device = cfg.device
    Ztr = Z_train.to(device)
    Zte = Z_test.to(device)

    # 1) train에서 seed 선택
    seed_idx = _sample_good_seeds_from_train(cost_train, cfg.good_q, cfg.num_seeds, seed=seed)
    S = len(seed_idx)
    seed_cost = cost_train[seed_idx].astype(np.float32)

    neigh_mean = np.zeros(S, dtype=np.float32)
    neigh_best = np.zeros(S, dtype=np.float32)
    rand_mean  = np.zeros(S, dtype=np.float32)
    rand_best  = np.zeros(S, dtype=np.float32)

    Nte, d = Zte.shape

    for si, tr_i in enumerate(seed_idx):
        z0 = Ztr[int(tr_i)]

        # --- neighborhood: train seed 주위에서 query 생성하지만, retrieval은 test pool ---
        c_list = []
        for _ in range(cfg.M):
            zq = _perturb_l2_ball(z0, cfg.eps)
            idxs, _ = retrieve_topk(
                z_query=zq,
                Z=Zte,
                k=cfg.retrieve_k,
                metric=cfg.retrieve_metric,
                exclude=None  # train seed는 test에 없으니 exclude 불필요
            )
            pick = int(to_numpy(idxs)[0])
            c_list.append(float(cost_test[pick]))
        c_arr = np.array(c_list, dtype=np.float32)
        neigh_mean[si] = float(c_arr.mean())
        neigh_best[si] = float(c_arr.max())

        # --- baseline: test에서 랜덤 query로 retrieval ---
        b_list = []
        for _ in range(cfg.M):
            if cfg.baseline_query_mode == "sample":
                ridx = random.randrange(Nte)
                zq = Zte[ridx]
            elif cfg.baseline_query_mode == "gauss":
                zq = torch.randn(d, device=device)
            else:
                raise ValueError(cfg.baseline_query_mode)

            idxs, _ = retrieve_topk(
                z_query=zq,
                Z=Zte,
                k=cfg.retrieve_k,
                metric=cfg.retrieve_metric,
                exclude=None
            )
            pick = int(to_numpy(idxs)[0])
            b_list.append(float(cost_test[pick]))
        b_arr = np.array(b_list, dtype=np.float32)
        rand_mean[si] = float(b_arr.mean())
        rand_best[si] = float(b_arr.max())

    out = {
        "seed_idx_train": seed_idx.astype(np.int32),
        "seed_cost_train": seed_cost,
        "neigh_cost_mean_test": neigh_mean,
        "neigh_cost_best_test": neigh_best,
        "rand_cost_mean_test": rand_mean,
        "rand_cost_best_test": rand_best,

        # gain은 "test 기준으로" 보는 게 자연스럽다: neigh - rand
        "gain_mean_test": (neigh_mean - rand_mean),
        "gain_best_test": (neigh_best - rand_best),
        "win_rate_mean": (neigh_mean > rand_mean).astype(np.int32),
        "win_rate_best": (neigh_best > rand_best).astype(np.int32),
    }
    return out

def summarize_train_seed_test_pool(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    nm = out["neigh_cost_mean_test"]
    nb = out["neigh_cost_best_test"]
    rm = out["rand_cost_mean_test"]
    rb = out["rand_cost_best_test"]

    return {
        "S": float(len(nm)),
        "neigh_mean_test": float(nm.mean()),
        "rand_mean_test": float(rm.mean()),
        "neigh_best_mean_test": float(nb.mean()),
        "rand_best_mean_test": float(rb.mean()),
        "win_rate_mean": float(out["win_rate_mean"].mean()),
        "win_rate_best": float(out["win_rate_best"].mean()),
        "avg_gain_mean_test": float((nm - rm).mean()),
        "avg_gain_best_test": float((nb - rb).mean()),
        "seed_mean_train": float(out["seed_cost_train"].mean()),
    }

import numpy as np

def plot_train_seed_test_pool(out: dict, title: str = "", bins: int = 30, save_path: str = None):
    """
    one_step_neighborhood_test_train_seed_test_pool() 출력(out)을 시각화.
    - test에서의 neigh vs random을 mean/best로 각각 히스토그램 비교
    - gain(neigh-rand) 분포도 같이 보여줌
    - win-rate / 평균 등 핵심 요약을 그래프 제목에 박아둠

    주의:
    - seed_cost_train은 train 분포라서 test 분포와 직접 비교 히스토그램에 섞지 않음
      (원하면 별도 subplot로 추가 가능)
    """
    import matplotlib.pyplot as plt

    nm = out["neigh_cost_mean_test"]
    rm = out["rand_cost_mean_test"]
    nb = out["neigh_cost_best_test"]
    rb = out["rand_cost_best_test"]

    gm = out["gain_mean_test"]
    gb = out["gain_best_test"]

    win_m = float(out["win_rate_mean"].mean())
    win_b = float(out["win_rate_best"].mean())

    # 간단 요약치
    def _q(a):
        return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))

    nm_med, nm_q1, nm_q3 = _q(nm)
    rm_med, rm_q1, rm_q3 = _q(rm)
    nb_med, nb_q1, nb_q3 = _q(nb)
    rb_med, rb_q1, rb_q3 = _q(rb)

    gm_med, gm_q1, gm_q3 = _q(gm)
    gb_med, gb_q1, gb_q3 = _q(gb)

    # --- Plot 1: Mean-of-M distributions ---
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(nm, bins=bins, alpha=0.5, label="neigh mean (test)")
    plt.hist(rm, bins=bins, alpha=0.5, label="random mean (test)")
    plt.axvline(nm_med, linestyle="--")
    plt.axvline(rm_med, linestyle="--")
    plt.title(
        (title + " | Mean-of-M (test)  "
         f"win={win_m:.3f}  "
         f"med(neigh)={nm_med:.3f}[{nm_q1:.3f},{nm_q3:.3f}]  "
         f"med(rand)={rm_med:.3f}[{rm_q1:.3f},{rm_q3:.3f}]")
    )
    plt.xlabel("true cost (bigger is better)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Best-of-M distributions ---
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(nb, bins=bins, alpha=0.5, label="neigh best-of-M (test)")
    plt.hist(rb, bins=bins, alpha=0.5, label="random best-of-M (test)")
    plt.axvline(nb_med, linestyle="--")
    plt.axvline(rb_med, linestyle="--")
    plt.title(
        (title + " | Best-of-M (test)  "
         f"win={win_b:.3f}  "
         f"med(neigh)={nb_med:.3f}[{nb_q1:.3f},{nb_q3:.3f}]  "
         f"med(rand)={rb_med:.3f}[{rb_q1:.3f},{rb_q3:.3f}]")
    )
    plt.xlabel("true cost (bigger is better)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Gain distributions (neigh - rand) ---
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(gm, bins=bins, alpha=0.6, label="gain mean (neigh - rand)")
    plt.axvline(0.0, linestyle="--")
    plt.axvline(gm_med, linestyle="--")
    plt.title(
        (title + " | Gain Mean-of-M (test)  "
         f"med={gm_med:.3f}[{gm_q1:.3f},{gm_q3:.3f}]  "
         f"mean={float(gm.mean()):.3f}")
    )
    plt.xlabel("gain (bigger is better)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8.5, 4.8))
    plt.hist(gb, bins=bins, alpha=0.6, label="gain best-of-M (neigh - rand)")
    plt.axvline(0.0, linestyle="--")
    plt.axvline(gb_med, linestyle="--")
    plt.title(
        (title + " | Gain Best-of-M (test)  "
         f"med={gb_med:.3f}[{gb_q1:.3f},{gb_q3:.3f}]  "
         f"mean={float(gb.mean()):.3f}")
    )
    plt.xlabel("gain (bigger is better)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)


def plot_seed_train_distribution(out: dict, title: str = "", bins: int = 30, save_path: str = None):
    """
    참고용: train seed cost 분포만 따로 그림.
    (test 분포와 직접 비교는 해석이 애매할 수 있어서 분리)
    """
    import matplotlib.pyplot as plt

    sc = out["seed_cost_train"]
    med = float(np.median(sc))
    q1 = float(np.quantile(sc, 0.25))
    q3 = float(np.quantile(sc, 0.75))

    plt.figure(figsize=(8.5, 4.2))
    plt.hist(sc, bins=bins, alpha=0.7)
    plt.axvline(med, linestyle="--")
    plt.title(title + f" | Train seed cost  med={med:.3f}[{q1:.3f},{q3:.3f}]")
    plt.xlabel("train seed true cost (bigger is better)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)



def plot_interpolation_search(vae, X_train, y_train, X_val, y_val, device, type="latent", save_path: str = None, only_summary: bool = False):
    # type : "latent" or "input"

    cfg = OneStepCfg(
        good_q=0.10,
        num_seeds=200,
        M=32,
        eps=0.3,
        retrieve_metric="l2",
        retrieve_k=1,
        baseline_query_mode="sample",
    )


    if type == "latent":
        vae.eval()
        with torch.no_grad():
            _, mu, _, z_train, _ = vae(torch.tensor(X_train).float().to(device))
            _, mu, _, z_val, _ = vae(torch.tensor(X_val).float().to(device))

        z_train = z_train.cpu()
        z_val = z_val.cpu()
        out = one_step_neighborhood_test_train_seed_test_pool(
            Z_train=z_train, cost_train=y_train,
            Z_test=z_val,   cost_test=y_val,
            cfg=cfg, seed=0
        )

    elif type == "input":
        X_train_tensor = torch.tensor(X_train).float()
        X_val_tensor = torch.tensor(X_val).float()

        out = one_step_neighborhood_test_train_seed_test_pool(
            Z_train=X_train_tensor, cost_train=y_train,
            Z_test=X_val_tensor,   cost_test=y_val,
            cfg=cfg, seed=0
        )

    
        
    elif type == "latent_all":
        vae.eval()
        with torch.no_grad():
            _, _, _, z_all, _ = vae(torch.tensor(np.concatenate([X_train, X_val], axis=0)).float().to(device))
        z_all = z_all.cpu()

        y_all = np.concatenate([y_train, y_val], axis=0)
        out = one_step_neighborhood_test_train_seed_test_pool(
            Z_train=z_all, cost_train=y_all,
            Z_test=z_all,   cost_test=y_all,
            cfg=cfg, seed=0
        )


    elif type == "input_all":
        X_all = np.concatenate([X_train, X_val], axis=0)
        X_all_tensor = torch.tensor(X_all).float()

        out = one_step_neighborhood_test_train_seed_test_pool(
            Z_train=X_all_tensor, cost_train=y_all,
            Z_test=X_all_tensor,   cost_test=y_all,
            cfg=cfg, seed=0
        )

    if only_summary:
        summary = summarize_train_seed_test_pool(out)
        # print(summary)
        return summary
    # print(summarize_train_seed_test_pool(out))
    plot_train_seed_test_pool(out, title=f"One-step (eps={cfg.eps}, M={cfg.M})", bins=30, save_path=save_path)
    plot_seed_train_distribution(out, title="Seeds", bins=30, save_path=save_path)
