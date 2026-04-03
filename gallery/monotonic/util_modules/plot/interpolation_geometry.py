"""
Latent interpolation 실험 (decode 없이, test set retrieval만 사용)

요구사항:
- test 데이터셋에 대해 encoder로 latent z를 뽑을 수 있어야 함
- 각 샘플의 true cost (측정치)가 있어야 함
- decode는 전혀 안 함
- 보간된 latent z(t)에 대해 test latent들 중 nearest(top-k) 를 찾아 그 cost로 곡선/통계를 봄

사용:
1) 아래 TODO 섹션의 `encode_testset()`만 너 환경에 맞게 채우면 됨
2) run_interpolation_experiment(...) 실행
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np
import torch

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def lerp(zA: torch.Tensor, zB: torch.Tensor, t: float) -> torch.Tensor:
    return (1.0 - t) * zA + t * zB

def slerp(zA: torch.Tensor, zB: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical linear interpolation.
    - 보통 z를 L2 normalize해서 쓰는 게 안정적.
    - z가 0 근처거나 A,B가 거의 동일하면 lerp로 fallback.
    """
    a = zA
    b = zB
    a_n = a / (a.norm() + eps)
    b_n = b / (b.norm() + eps)
    dot = torch.clamp(torch.dot(a_n, b_n), -1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    if torch.isnan(omega) or omega.abs() < 1e-5:
        return lerp(a, b, t)
    so = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

def pairwise_dist_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [d], b: [N,d] -> [N]
    return ((b - a.unsqueeze(0)) ** 2).sum(dim=-1)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # a: [d], b: [N,d] -> [N]
    a_n = a / (a.norm() + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return torch.matmul(b_n, a_n)

# -----------------------------
# Retrieval (pure torch, no faiss)
# -----------------------------

def retrieve_topk(
    z_query: torch.Tensor,
    Z: torch.Tensor,
    k: int = 10,
    metric: str = "l2",
    exclude: Optional[set] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (idxs, dists_or_sims)
    - metric="l2": smaller is better, returns squared l2 distances
    - metric="cos": larger is better, returns cosine similarities
    """
    if exclude is None:
        exclude = set()

    if metric == "l2":
        d = pairwise_dist_l2(z_query, Z)  # [N]
        # mask excludes by setting +inf
        if len(exclude) > 0:
            mask = torch.zeros_like(d, dtype=torch.bool)
            ex = torch.tensor(list(exclude), device=d.device, dtype=torch.long)
            mask[ex] = True
            d = d.masked_fill(mask, float("inf"))
        vals, idxs = torch.topk(d, k=min(k, Z.shape[0]), largest=False)
        return idxs, vals
    elif metric == "cos":
        s = cosine_sim(z_query, Z)  # [N]
        if len(exclude) > 0:
            mask = torch.zeros_like(s, dtype=torch.bool)
            ex = torch.tensor(list(exclude), device=s.device, dtype=torch.long)
            mask[ex] = True
            s = s.masked_fill(mask, float("-inf"))
        vals, idxs = torch.topk(s, k=min(k, Z.shape[0]), largest=True)
        return idxs, vals
    else:
        raise ValueError(f"Unknown metric: {metric}")

# -----------------------------
# Pair sampling
# -----------------------------

@dataclass
class PairSamplingConfig:
    good_q: float = 0.10   # 상위 10% = good (cost 낮은 쪽)
    bad_q: float = 0.10    # 하위 10% = bad (cost 높은 쪽)
    min_latent_dist_quantile: float = 0.75  # "서로 먼 pair" 필터용 (거리 상위 25%)
    metric: str = "l2"     # pair distance 기준
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def quantile_thresholds(cost: np.ndarray, good_q: float, bad_q: float) -> Tuple[float, float]:
    # cost(=neg log) 클수록 좋다고 가정 (maximize)
    good_thr = np.quantile(cost, 1.0 - good_q)  # 높은 쪽
    bad_thr  = np.quantile(cost, bad_q)         # 낮은 쪽
    return float(good_thr), float(bad_thr)


def sample_pairs(
    Z: torch.Tensor,
    cost: np.ndarray,
    num_pairs_each: int,
    cfg: PairSamplingConfig,
    seed: int = 0,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Returns dict with keys: "good_good", "good_bad", "bad_bad"
    Pair는 (i, j) index
    """
    set_seed(seed)
    N, d = Z.shape
    good_thr, bad_thr = quantile_thresholds(cost, cfg.good_q, cfg.bad_q)

    good_idx = np.where(cost >= good_thr)[0]  # good: 큰 값
    bad_idx  = np.where(cost <= bad_thr)[0]   # bad: 작은 값


    if len(good_idx) < 2 or len(bad_idx) < 2:
        raise RuntimeError("good/bad 샘플이 너무 적음. quantile을 조절해라.")

    # 빠른 샘플링을 위해 torch 텐서로
    Zt = Z.to(cfg.device)

    def latent_distance(i: int, j: int) -> float:
        zi = Zt[i]
        zj = Zt[j]
        if cfg.metric == "l2":
            return float(((zi - zj) ** 2).sum().detach().cpu().item())
        elif cfg.metric == "cos":
            return float((zi * zj).sum().detach().cpu().item() / (zi.norm().cpu().item() * zj.norm().cpu().item() + 1e-12))
        else:
            raise ValueError(cfg.metric)

    # 거리 분포를 대충 추정해서 "먼 pair" threshold 잡기
    # (정확히 다 계산하면 O(N^2)라 미친짓)
    probe = 5000
    ds = []
    for _ in range(min(probe, N * 10)):
        i = random.randrange(N)
        j = random.randrange(N)
        if i == j:
            continue
        if cfg.metric == "l2":
            ds.append(latent_distance(i, j))
        else:
            # cos이면 "멀다"를 (1-cos)로 치환해서 비교
            ds.append(1.0 - latent_distance(i, j))
    dist_thr = float(np.quantile(ds, cfg.min_latent_dist_quantile))

    def is_far(i: int, j: int) -> bool:
        if cfg.metric == "l2":
            return latent_distance(i, j) >= dist_thr
        else:
            # cos case: ds는 (1-cos)이었음
            c = latent_distance(i, j)
            return (1.0 - c) >= dist_thr

    def draw_from(pool_a: np.ndarray, pool_b: np.ndarray, need: int, same_pool: bool) -> List[Tuple[int, int]]:
        pairs = []
        trials = 0
        max_trials = need * 200
        while len(pairs) < need and trials < max_trials:
            trials += 1
            i = int(random.choice(pool_a))
            j = int(random.choice(pool_b))
            if i == j:
                continue
            if same_pool and i > j:
                i, j = j, i
            if not is_far(i, j):
                continue
            pairs.append((i, j))
        if len(pairs) < need:
            print(f"[warn] pairs 부족: {len(pairs)}/{need}. dist quantile을 낮춰라(예: 0.6).")
        return pairs

    out = {
        "good_good": draw_from(good_idx, good_idx, num_pairs_each, same_pool=True),
        "good_bad":  draw_from(good_idx, bad_idx,  num_pairs_each, same_pool=False),
        "bad_bad":   draw_from(bad_idx,  bad_idx,   num_pairs_each, same_pool=True),
    }
    return out

# -----------------------------
# Metrics on curves
# -----------------------------

@dataclass
class CurveMetrics:
    total_variation: float
    monotone_violations: int
    spearman_corr: float
    unique_programs: int

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # 간단 구현 (scipy 없이)
    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a), dtype=float)
        # tie 처리: 간단히 평균 랭크로 (대충)
        # tie 많지 않다고 가정하거나, 중요하면 scipy 쓰자.
        return ranks
    rx = rankdata(x)
    ry = rankdata(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = (np.sqrt((vx**2).sum()) * np.sqrt((vy**2).sum()) + 1e-12)
    return float((vx * vy).sum() / denom)

def compute_curve_metrics(t_vals: np.ndarray, cost_vals: np.ndarray, picked_ids: List[int], expect_increasing: Optional[bool]) -> CurveMetrics:
    tv = float(np.abs(np.diff(cost_vals)).sum())
    # monotone violations
    viol = 0
    if expect_increasing is not None:
        diffs = np.diff(cost_vals)
        if expect_increasing:
            viol = int((diffs < 0).sum())
        else:
            viol = int((diffs > 0).sum())

    sp = spearman_corr(t_vals, cost_vals)
    uniq = len(set(picked_ids))
    return CurveMetrics(total_variation=tv, monotone_violations=viol, spearman_corr=sp, unique_programs=uniq)

# -----------------------------
# Main experiment
# -----------------------------

@dataclass
class InterpConfig:
    t_grid: List[float] = None
    interp: str = "lerp"         # "lerp" or "slerp"
    retrieve_k: int = 10
    retrieve_metric: str = "l2"  # "l2" or "cos"
    exclude_anchors: bool = True
    pick_mode: str = "median"    # "nn" or "median"
    use_log_cost: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.t_grid is None:
            self.t_grid = [i / 10.0 for i in range(11)]  # 0.0..1.0

def run_one_pair_path(
    i: int, j: int,
    Z: torch.Tensor,
    cost: np.ndarray,
    icfg: InterpConfig,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Returns:
    - path_cost: [T] (picked cost per t)
    - picked_ids: list length T (which program chosen per t)
    - path_cost_kstats: [T] (top-k median/mean cost per t depending on pick_mode)
    """
    Zt = Z.to(icfg.device)
    zA = Zt[i]
    zB = Zt[j]

    t_vals = np.array(icfg.t_grid, dtype=np.float32)
    path_cost = []
    picked_ids = []
    path_kstat = []

    exclude = set()
    if icfg.exclude_anchors:
        exclude = {i, j}

    for t in t_vals:
        if icfg.interp == "lerp":
            zt = lerp(zA, zB, float(t))
        elif icfg.interp == "slerp":
            zt = slerp(zA, zB, float(t))
        else:
            raise ValueError(icfg.interp)

        idxs, vals = retrieve_topk(
            z_query=zt,
            Z=Zt,
            k=icfg.retrieve_k,
            metric=icfg.retrieve_metric,
            exclude=exclude
        )
        idxs_np = to_numpy(idxs).astype(int)

        # top-k의 cost 통계
        costs_k = cost[idxs_np]
        if icfg.use_log_cost:
            costs_k = np.log(costs_k + 1e-12)

        if icfg.pick_mode == "nn":
            pick = idxs_np[0]
            stat = float(costs_k[0])
        elif icfg.pick_mode == "median":
            # 가장 가까운 하나는 튈 수 있으니, top-k median을 본다
            stat = float(np.median(costs_k))
            # path를 대표하는 picked id는 "top-k 중 cost가 median에 가장 가까운 것"으로 선택(그럴듯)
            pick = int(idxs_np[np.argmin(np.abs(costs_k - stat))])
        elif icfg.pick_mode == "mean":
            stat = float(np.mean(costs_k))
            pick = int(idxs_np[np.argmin(np.abs(costs_k - stat))])
        else:
            raise ValueError(icfg.pick_mode)

        path_cost.append(float(stat))
        picked_ids.append(pick)
        path_kstat.append(float(stat))

    return np.array(path_cost, dtype=np.float32), picked_ids, np.array(path_kstat, dtype=np.float32)

def run_interpolation_experiment(
    Z_test: torch.Tensor,
    true_cost: np.ndarray,
    num_pairs_each: int = 200,
    seed: int = 0,
    pair_cfg: Optional[PairSamplingConfig] = None,
    interp_cfg: Optional[InterpConfig] = None,
) -> Dict[str, Dict]:
    """
    Runs interpolation experiment for 3 pair types, returns summary dict.

    Output structure:
    {
      "good_good": {
         "curves": [T arrays...],
         "metrics": [CurveMetrics...],
         "t": t_grid,
      },
      ...
    }
    """
    assert isinstance(true_cost, np.ndarray)
    assert len(true_cost.shape) == 1
    assert Z_test.shape[0] == true_cost.shape[0]

    if pair_cfg is None:
        pair_cfg = PairSamplingConfig()
    if interp_cfg is None:
        interp_cfg = InterpConfig()

    pairs_by_type = sample_pairs(Z_test, true_cost, num_pairs_each, pair_cfg, seed=seed)

    results: Dict[str, Dict] = {}
    t_vals = np.array(interp_cfg.t_grid, dtype=np.float32)

    for ptype, pairs in pairs_by_type.items():
        curves = []
        kstats = []
        metrics = []
        picked_paths = []

        # monotonic expectation:
        # - good_bad: t가 0->1로 갈수록 보통 cost가 증가(나빠짐) 기대
        # - good_good / bad_bad: 단조 기대 없음
        expect_increasing = False if ptype == "good_bad" else None


        for (i, j) in pairs:
            curve, picked_ids, kstat = run_one_pair_path(i, j, Z_test, true_cost, interp_cfg)
            curves.append(curve)
            kstats.append(kstat)
            picked_paths.append(picked_ids)

            m = compute_curve_metrics(t_vals, curve, picked_ids, expect_increasing=expect_increasing)
            metrics.append(m)

        # aggregate stats
        C = np.stack(curves, axis=0) if len(curves) > 0 else np.zeros((0, len(t_vals)), dtype=np.float32)

        results[ptype] = {
            "t": t_vals,
            "curves": C,                # [P, T]
            "metrics": metrics,         # list of CurveMetrics
            "picked_paths": picked_paths,
            "summary": summarize_metrics(metrics, ptype),
        }

    return results

def summarize_metrics(metrics: List[CurveMetrics], ptype: str) -> Dict[str, float]:
    if len(metrics) == 0:
        return {}
    tv = np.array([m.total_variation for m in metrics], dtype=np.float32)
    viol = np.array([m.monotone_violations for m in metrics], dtype=np.float32)
    sp = np.array([m.spearman_corr for m in metrics], dtype=np.float32)
    uniq = np.array([m.unique_programs for m in metrics], dtype=np.float32)

    out = {
        "num_pairs": round(float(len(metrics)), 4),
        "tv_mean": round(float(tv.mean()), 4),
        "tv_median": round(float(np.median(tv)), 4),
        "spearman_mean": round(float(sp.mean()), 4),
        "spearman_median": round(float(np.median(sp)), 4),
        "unique_mean": round(float(uniq.mean()), 4),
        "unique_median": round(float(np.median(uniq)), 4),
    }
    if ptype == "good_bad":
        out.update({
            "mono_viol_mean": round(float(viol.mean()), 4),
            "mono_viol_median": round(float(np.median(viol)), 4),
        })
    return out

# -----------------------------
# Baselines
# -----------------------------

def random_baseline_curves(true_cost: np.ndarray, t_grid: List[float], num_curves: int, k: int = 10, seed: int = 0, use_log_cost: bool = True):
    """
    각 t마다 test에서 랜덤 k개 뽑아 median cost를 curve로 만드는 baseline.
    """
    set_seed(seed)
    N = true_cost.shape[0]
    t_vals = np.array(t_grid, dtype=np.float32)
    curves = []
    for _ in range(num_curves):
        curve = []
        for _t in t_vals:
            idxs = np.random.choice(N, size=min(k, N), replace=False)
            c = true_cost[idxs]
            if use_log_cost:
                c = -np.log(c + 1e-12)
            curve.append(float(np.median(c)))
        curves.append(np.array(curve, dtype=np.float32))
    return t_vals, np.stack(curves, axis=0)

# -----------------------------
# Plotting (optional)
# -----------------------------

def plot_aggregate(results: Dict[str, Dict], baseline: Optional[Tuple[np.ndarray, np.ndarray]] = None, title_prefix: str = "", save_path: Optional[str] = None):
    """
    matplotlib만 사용. (색 지정 안 함)
    - 각 ptype 별로 t에 따른 median curve와 IQR을 그려줌.
    """
    import matplotlib.pyplot as plt

    for ptype, r in results.items():
        t = r["t"]
        C = r["curves"]
        if C.shape[0] == 0:
            continue
        med = np.median(C, axis=0)
        q1 = np.quantile(C, 0.25, axis=0)
        q3 = np.quantile(C, 0.75, axis=0)

        plt.figure(figsize=(8, 4.5))
        plt.plot(t, med, linewidth=2, label=f"{ptype} median")
        plt.fill_between(t, q1, q3, alpha=0.25, label="IQR")

        if baseline is not None:
            tb, Cb = baseline
            med_b = np.median(Cb, axis=0)
            q1b = np.quantile(Cb, 0.25, axis=0)
            q3b = np.quantile(Cb, 0.75, axis=0)
            plt.plot(tb, med_b, linewidth=2, linestyle="--", label="random baseline median")
            plt.fill_between(tb, q1b, q3b, alpha=0.15, label="baseline IQR")

        plt.xlabel("t (interpolation)")
        plt.ylabel("log(cost)" if True else "cost")
        plt.title(f"{title_prefix}{ptype}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

# -----------------------------
# TODO: 너 환경에 맞게 채워야 하는 부분
# -----------------------------

def encode_testset() -> Tuple[torch.Tensor, np.ndarray]:
    """
    여기만 너 코드로 바꿔.
    반환:
      Z_test: torch.Tensor [N, d]
      true_cost: np.ndarray [N]  (cost 낮을수록 좋다고 가정)

    예시(가짜):
      - 너는 보통 dataloader로 x를 가져오고
      - encoder(x) -> z_mean 또는 z_sample을 쓸거야
      - decode는 필요 없음
    """
    raise NotImplementedError("encode_testset()을 너 환경에 맞게 구현해라.")

# -----------------------------
# Example main
# -----------------------------


def plot_interpolation_geometry(vae, X_train, y_train, X_val, y_val, device, type="latent", save_path: str = None, only_summary: bool = False):

    if type == "latent":
        vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar, z, cost_pred = vae(torch.tensor(X_val).float().to(device))



    # 1) pair 샘플링 설정
    pair_cfg = PairSamplingConfig(
        good_q=0.10,
        bad_q=0.10,
        min_latent_dist_quantile=0.75,
        metric="l2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2) interpolation + retrieval 설정
    interp_cfg = InterpConfig(
        t_grid=[i/10 for i in range(11)],
        interp="lerp",          # "slerp"도 돌려봐
        retrieve_k=10,
        retrieve_metric="l2",   # "cos"로 바꾸면 Z를 normalize해 쓰는 게 좋음
        exclude_anchors=True,
        pick_mode="median",     # "nn" / "median" / "mean"
        use_log_cost=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3) 실행
    if type == "latent":
        results = run_interpolation_experiment(
            Z_test=z,
            true_cost=y_val,
            num_pairs_each=200,
            seed=0,
            pair_cfg=pair_cfg,
            interp_cfg=interp_cfg
        )
    elif type == "input":
        results = run_interpolation_experiment(
            Z_test=torch.tensor(X_val).float().to(interp_cfg.device),
            true_cost=y_val,
            num_pairs_each=200,
            seed=0,
            pair_cfg=pair_cfg,
            interp_cfg=interp_cfg
        )

    # print("=== Summary ===")
    summary = {}
    if only_summary:
        summary = {k: v["summary"] for k, v in results.items()}
        # print(summary)
        return summary

    # for k, v in results.items():
    #     print(k, v["summary"])

    # 4) baseline + plot
    tb, Cb = random_baseline_curves(y_val, interp_cfg.t_grid, num_curves=200, k=interp_cfg.retrieve_k, seed=0, use_log_cost=False)
    plot_aggregate(results, baseline=(tb, Cb), title_prefix="Interp Retrieval | ", save_path=save_path)
