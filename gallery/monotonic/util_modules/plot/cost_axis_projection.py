import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from sklearn.linear_model import Ridge


def fit_cost_direction(X_train, y_train, alpha=1.0):
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    w = reg.coef_
    w = w / (np.linalg.norm(w) + 1e-12)
    return w


def unit(v, eps=1e-12):
    return v / (np.linalg.norm(v) + eps)

def project_1d(Z, v):
    return Z @ unit(v)

def plot_monotonicity_error_scatter(Z, y, v, expect="increasing", stride=1, title_prefix="", show=True):
    """
    x축: t(축 projection)
    y축: cost (aligned)
    색: 단조성 위반 정도 (rank error)
        - 파랑: 단조성 잘 맞음
        - 빨강: 단조성 크게 위반
    """

    t = project_1d(Z, v)
    y = np.asarray(y)

    # 방향 맞추기 (increasing 기준)
    y_plot = -y if expect == "decreasing" else y

    # rank 계산
    rt = np.argsort(np.argsort(t))         # 0..N-1
    ry = np.argsort(np.argsort(y_plot))    # 0..N-1

    # 점별 "단조성 오류" (0~1로 정규화)
    err = np.abs(ry - rt).astype(np.float32)
    err = err / (err.max() + 1e-12)

    # y_val의 top-10 찾기 (원래 y 기준)
    top10_indices = np.argsort(y)[-10:]  # 가장 높은 10개 인덱스
    top1_index = np.argsort(y)[-1]     # 가장 높은 1개 인덱스

    # 정렬(보기 좋게)
    order = np.argsort(t)
    t_s = t[order][::stride]
    y_s = y_plot[order][::stride]
    err_s = err[order][::stride]
    
    # 정렬된 인덱스에서 top-5 찾기
    original_indices = order[::stride]
    
    # 전체 단조성 척도 (제목에)
    rho = float(spearmanr(t, y_plot).correlation)
    tau = float(kendalltau(t, y_plot).correlation)

    plt.figure(figsize=(9, 4))
    
    # 일반 점들
    sc = plt.scatter(t_s, y_s, c=err_s, s=12, alpha=0.7)
    
    # Top-10 중 2-10위 (주황색)
    top10_mask = np.isin(original_indices, top10_indices)
    top1_mask = np.isin(original_indices, [top1_index])
    top2to10_mask = top10_mask & ~top1_mask
    
    if np.any(top2to10_mask):
        plt.scatter(t_s[top2to10_mask], y_s[top2to10_mask], c='orange', s=20, alpha=0.9, label='Top 2-10')
    
    # Top-1 (빨간색)
    if np.any(top1_mask):
        plt.scatter(t_s[top1_mask], y_s[top1_mask], c='red', s=20, alpha=1.0, label='Top 1')
    
    # cost 축 기준 수평선 추가
    max_cost_val = y_plot.max()  # 최고 cost 값 (aligned) 전체에서
    top1_cost_val = y_plot[top1_index] if len(y_plot) > top1_index else None
    
    # v축(t축) 기준으로 맨 끝에 있는 점의 cost 값
    max_t_index = np.argmax(t)  # t값이 가장 큰 점의 인덱스
    max_t_val = t[max_t_index]
    max_t_cost_val = y_plot[max_t_index]  # v축 맨 끝 점의 cost 값
    
    # 보라색 수평선 (v축 맨 끝 점의 cost)
    plt.axhline(y=max_t_cost_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Max t-axis cost: {max_t_cost_val:.2f}')
    # 보라색 수직선
    plt.axvline(x=max_t_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6)
    
    # 빨간색 수평선 (top-1 cost)
    if top1_cost_val is not None:
        plt.axhline(y=top1_cost_val, color='red', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Top-1 cost: {top1_cost_val:.2f}')
        plt.axvline(x=t[top1_index], color='red', linestyle='--', alpha=0.8, linewidth=0.6)
    
    if show:
        plt.colorbar(sc, label="monotonicity error (blue=good, red=bad)")
        plt.xlabel("t = projection onto v")
        plt.ylabel("cost (aligned)")
        plt.title(f"{title_prefix}Monotonicity error colored")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Top-1 cost value (aligned): {top1_cost_val:.4f}")
    # top-1이 v축 기준으로 맨 끝에서 몇 번째인지
    top1_t_rank = np.argsort(np.argsort(t))[top1_index]
    print(f"Top-1 t-axis rank: {len(t)-top1_t_rank}")
    
    print(f"Max t-axis cost value (aligned): {max_t_cost_val:.4f}")
    print(f"Max t value: {t.max():.4f}")

    print(f"t-axis Top-10 mean : {y_plot[np.argsort(t)[-10:]].mean():.4f}")
    print(f"spearman_rho : {rho:.4f}")
    print(f"kendall_tau : {tau:.4f}")

    if not show:
        plt.close()
        results = {
            "true max rank": len(t)-top1_t_rank,
            "true max cost": round(top1_cost_val, 4),
            "max cost": round(max_t_cost_val, 4),
            "top-10 mean": round(y_plot[np.argsort(t)[-10:]].mean(), 4),
            "spearman_rho": rho,
            "kendall_tau": tau,
        }
        return results
    




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

def plot_monotonicity_error_scatter_slerp(Z, y, w1, w2, expect="increasing", stride=1, title_prefix="", show=True):
    """
    x축: theta (w1, w2 평면 상의 각도, -pi ~ pi)
    y축: cost (aligned)
    색: 단조성 위반 정도 (rank error)
        - 파랑: 단조성 잘 맞음
        - 빨강: 단조성 크게 위반
    """
    Z = np.asarray(Z)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    y = np.asarray(y)

    # 1. w1, w2 직교화 및 정규화 (모델과 동일한 로직)
    w1_norm = w1 / (np.linalg.norm(w1) + 1e-12)
    proj = np.dot(w2, w1_norm) * w1_norm
    w2_ortho = w2 - proj
    w2_norm = w2_ortho / (np.linalg.norm(w2_ortho) + 1e-12)

    # 2. 2D 평면 투영 및 각도(theta) 계산
    u = Z @ w1_norm
    v = Z @ w2_norm
    theta = np.arctan2(v, u)  # 범위: [-pi, pi]

    # 방향 맞추기 (increasing 기준)
    y_plot = -y if expect == "decreasing" else y

    # rank 계산 (t 대신 theta 사용)
    rt = np.argsort(np.argsort(theta))         # 0..N-1
    ry = np.argsort(np.argsort(y_plot))    # 0..N-1

    # 점별 "단조성 오류" (0~1로 정규화)
    err = np.abs(ry - rt).astype(np.float32)
    err = err / (err.max() + 1e-12)

    # y_val의 top-10 찾기 (원래 y 기준)
    top10_indices = np.argsort(y)[-10:]  # 가장 높은 10개 인덱스
    top1_index = np.argsort(y)[-1]     # 가장 높은 1개 인덱스

    # 정렬(보기 좋게)
    order = np.argsort(theta)
    theta_s = theta[order][::stride]
    y_s = y_plot[order][::stride]
    err_s = err[order][::stride]
    
    # 정렬된 인덱스에서 top-5 찾기
    original_indices = order[::stride]
    
    # 전체 단조성 척도 (제목에)
    rho = float(spearmanr(theta, y_plot).correlation)
    tau = float(kendalltau(theta, y_plot).correlation)

    plt.figure(figsize=(9, 4))
    
    # 일반 점들
    sc = plt.scatter(theta_s, y_s, c=err_s, s=12, alpha=0.7, cmap='coolwarm')
    
    # Top-10 중 2-10위 (주황색)
    top10_mask = np.isin(original_indices, top10_indices)
    top1_mask = np.isin(original_indices, [top1_index])
    top2to10_mask = top10_mask & ~top1_mask
    
    if np.any(top2to10_mask):
        plt.scatter(theta_s[top2to10_mask], y_s[top2to10_mask], c='orange', s=20, alpha=0.9, label='Top 2-10')
    
    # Top-1 (빨간색)
    if np.any(top1_mask):
        plt.scatter(theta_s[top1_mask], y_s[top1_mask], c='red', s=20, alpha=1.0, label='Top 1')
    
    # cost 축 기준 수평선 추가
    max_cost_val = y_plot.max()  # 최고 cost 값 (aligned) 전체에서
    top1_cost_val = y_plot[top1_index] if len(y_plot) > top1_index else None
    
    # theta축 기준으로 맨 끝에 있는 점의 cost 값 (theta가 가장 큰 점)
    max_theta_index = np.argmax(theta)  
    max_theta_val = theta[max_theta_index]
    max_theta_cost_val = y_plot[max_theta_index]  
    
    # 보라색(초록색) 수평/수직선 (theta축 맨 끝 점의 cost)
    plt.axhline(y=max_theta_cost_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Max angle cost: {max_theta_cost_val:.2f}')
    plt.axvline(x=max_theta_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6)
    
    # 빨간색 수평/수직선 (top-1 cost)
    if top1_cost_val is not None:
        plt.axhline(y=top1_cost_val, color='red', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Top-1 cost: {top1_cost_val:.2f}')
        plt.axvline(x=theta[top1_index], color='red', linestyle='--', alpha=0.8, linewidth=0.6)
    
    if show:
        plt.colorbar(sc, label="monotonicity error (blue=good, red=bad)")
        plt.xlabel("theta (Angle on 2D plane)")
        plt.ylabel("cost (aligned)")
        plt.title(f"{title_prefix}Monotonicity error colored (SLERP)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Top-1 cost value (aligned): {top1_cost_val:.4f}")
    # top-1이 theta축 기준으로 맨 끝에서 몇 번째인지
    top1_theta_rank = np.argsort(np.argsort(theta))[top1_index]
    print(f"Top-1 theta-axis rank: {len(theta)-top1_theta_rank}")
    
    print(f"Max theta-axis cost value (aligned): {max_theta_cost_val:.4f}")
    print(f"Max theta value: {theta.max():.4f}")

    print(f"theta-axis Top-10 mean : {y_plot[np.argsort(theta)[-10:]].mean():.4f}")
    print(f"spearman_rho : {rho:.4f}")
    print(f"kendall_tau : {tau:.4f}")

    if not show:
        plt.close()
        results = {
            "true max rank": len(theta)-top1_theta_rank,
            "true max cost": round(top1_cost_val, 4),
            "max cost": round(max_theta_cost_val, 4),
            "top-10 mean": round(y_plot[np.argsort(theta)[-10:]].mean(), 4),
            "spearman_rho": rho,
            "kendall_tau": tau,
        }
        return results
    



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

def plot_monotonicity_error_scatter_explicit(Z, y, cost_dim=0, expect="increasing", stride=1, title_prefix="", show=True):
    """
    x축: t (명시적으로 할당된 잠재 공간의 특정 차원, 예: Z[:, 0])
    y축: cost (aligned)
    색: 단조성 위반 정도 (rank error)
        - 파랑: 단조성 잘 맞음
        - 빨강: 단조성 크게 위반
    """
    Z = np.asarray(Z)
    y = np.asarray(y)

    # 투영(Projection) 대신 명시적으로 할당된 차원의 값을 그대로 가져옵니다.
    t = Z[:, cost_dim]

    # 방향 맞추기 (increasing 기준)
    y_plot = -y if expect == "decreasing" else y

    # rank 계산
    rt = np.argsort(np.argsort(t))         # 0..N-1
    ry = np.argsort(np.argsort(y_plot))    # 0..N-1

    # 점별 "단조성 오류" (0~1로 정규화)
    err = np.abs(ry - rt).astype(np.float32)
    err = err / (err.max() + 1e-12)

    # y_val의 top-10 찾기 (원래 y 기준)
    top10_indices = np.argsort(y)[-10:]  # 가장 높은 10개 인덱스
    top1_index = np.argsort(y)[-1]     # 가장 높은 1개 인덱스

    # 정렬(보기 좋게)
    order = np.argsort(t)
    t_s = t[order][::stride]
    y_s = y_plot[order][::stride]
    err_s = err[order][::stride]
    
    # 정렬된 인덱스에서 top-5 찾기
    original_indices = order[::stride]
    
    # 전체 단조성 척도 (제목에)
    rho = float(spearmanr(t, y_plot).correlation)
    tau = float(kendalltau(t, y_plot).correlation)

    plt.figure(figsize=(9, 4))
    
    # 일반 점들
    sc = plt.scatter(t_s, y_s, c=err_s, s=12, alpha=0.7, cmap='coolwarm')
    
    # Top-10 중 2-10위 (주황색)
    top10_mask = np.isin(original_indices, top10_indices)
    top1_mask = np.isin(original_indices, [top1_index])
    top2to10_mask = top10_mask & ~top1_mask
    
    if np.any(top2to10_mask):
        plt.scatter(t_s[top2to10_mask], y_s[top2to10_mask], c='orange', s=20, alpha=0.9, label='Top 2-10')
    
    # Top-1 (빨간색)
    if np.any(top1_mask):
        plt.scatter(t_s[top1_mask], y_s[top1_mask], c='red', s=20, alpha=1.0, label='Top 1')
    
    # cost 축 기준 수평선 추가
    max_cost_val = y_plot.max()  # 최고 cost 값 (aligned) 전체에서
    top1_cost_val = y_plot[top1_index] if len(y_plot) > top1_index else None
    
    # 명시적 차원(t축) 기준으로 맨 끝에 있는 점의 cost 값
    max_t_index = np.argmax(t)  # t값이 가장 큰 점의 인덱스
    max_t_val = t[max_t_index]
    max_t_cost_val = y_plot[max_t_index]  # t축 맨 끝 점의 cost 값
    
    # 보라색(초록색) 수평선/수직선 (t축 맨 끝 점의 cost)
    plt.axhline(y=max_t_cost_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Max Z[{cost_dim}] cost: {max_t_cost_val:.2f}')
    plt.axvline(x=max_t_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6)
    
    # 빨간색 수평선/수직선 (top-1 cost)
    if top1_cost_val is not None:
        plt.axhline(y=top1_cost_val, color='red', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Top-1 cost: {top1_cost_val:.2f}')
        plt.axvline(x=t[top1_index], color='red', linestyle='--', alpha=0.8, linewidth=0.6)
    
    if show:
        plt.colorbar(sc, label="monotonicity error (blue=good, red=bad)")
        plt.xlabel(f"t = Z[:, {cost_dim}] (Explicit Cost Dimension)")
        plt.ylabel("cost (aligned)")
        plt.title(f"{title_prefix}Monotonicity error colored (Explicit Dim)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Top-1 cost value (aligned): {top1_cost_val:.4f}")
    # top-1이 t축 기준으로 맨 끝에서 몇 번째인지
    top1_t_rank = np.argsort(np.argsort(t))[top1_index]
    print(f"Top-1 Z-axis rank: {len(t)-top1_t_rank}")
    
    print(f"Max Z-axis cost value (aligned): {max_t_cost_val:.4f}")
    print(f"Max Z[{cost_dim}] value: {t.max():.4f}")

    print(f"Z-axis Top-10 mean : {y_plot[np.argsort(t)[-10:]].mean():.4f}")
    print(f"spearman_rho : {rho:.4f}")
    print(f"kendall_tau : {tau:.4f}")

    if not show:
        plt.close()
        results = {
            "true max rank": len(t)-top1_t_rank,
            "true max cost": round(top1_cost_val, 4),
            "max cost": round(max_t_cost_val, 4),
            "top-10 mean": round(y_plot[np.argsort(t)[-10:]].mean(), 4),
            "spearman_rho": rho,
            "kendall_tau": tau,
        }
        return results
    

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

def plot_monotonicity_error_scatter_curved(t, y, expect="increasing", stride=1, title_prefix="", show=True):
    """
    t축: 모델의 t_extractor가 예측한 곡선 상의 위치(진행도)
    y축: cost (aligned)
    색: 단조성 위반 정도 (rank error)
        - 파랑: 단조성 잘 맞음
        - 빨강: 단조성 크게 위반
    """
    
    # Z와 v를 받아 project_1d를 하던 부분을 삭제하고,
    # 입력받은 t를 바로 1차원 배열로 사용합니다.
    t = np.asarray(t).flatten()
    y = np.asarray(y)

    # 방향 맞추기 (increasing 기준)
    y_plot = -y if expect == "decreasing" else y

    # rank 계산
    rt = np.argsort(np.argsort(t))         # 0..N-1
    ry = np.argsort(np.argsort(y_plot))    # 0..N-1

    # 점별 "단조성 오류" (0~1로 정규화)
    err = np.abs(ry - rt).astype(np.float32)
    err = err / (err.max() + 1e-12)

    # y_val의 top-10 찾기 (원래 y 기준)
    top10_indices = np.argsort(y)[-10:]  # 가장 높은 10개 인덱스
    top1_index = np.argsort(y)[-1]     # 가장 높은 1개 인덱스

    # 정렬(보기 좋게)
    order = np.argsort(t)
    t_s = t[order][::stride]
    y_s = y_plot[order][::stride]
    err_s = err[order][::stride]
    
    # 정렬된 인덱스에서 top-5 찾기
    original_indices = order[::stride]
    
    # 전체 단조성 척도 (제목에)
    rho = float(spearmanr(t, y_plot).correlation)
    tau = float(kendalltau(t, y_plot).correlation)

    plt.figure(figsize=(9, 4))
    
    # 일반 점들
    sc = plt.scatter(t_s, y_s, c=err_s, s=12, alpha=0.7, cmap='coolwarm')
    
    # Top-10 중 2-10위 (주황색)
    top10_mask = np.isin(original_indices, top10_indices)
    top1_mask = np.isin(original_indices, [top1_index])
    top2to10_mask = top10_mask & ~top1_mask
    
    if np.any(top2to10_mask):
        plt.scatter(t_s[top2to10_mask], y_s[top2to10_mask], c='orange', s=20, alpha=0.9, label='Top 2-10')
    
    # Top-1 (빨간색)
    if np.any(top1_mask):
        plt.scatter(t_s[top1_mask], y_s[top1_mask], c='red', s=20, alpha=1.0, label='Top 1')
    
    # cost 축 기준 수평선 추가
    max_cost_val = y_plot.max()  # 최고 cost 값 (aligned) 전체에서
    top1_cost_val = y_plot[top1_index] if len(y_plot) > top1_index else None
    
    # t축 기준으로 맨 끝에 있는 점의 cost 값
    max_t_index = np.argmax(t)  # t값이 가장 큰 점의 인덱스
    max_t_val = t[max_t_index]
    max_t_cost_val = y_plot[max_t_index]  # t축 맨 끝 점의 cost 값
    
    # 보라색(초록색) 수평선/수직선 (t축 맨 끝 점의 cost)
    plt.axhline(y=max_t_cost_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Max curve-t cost: {max_t_cost_val:.2f}')
    plt.axvline(x=max_t_val, color='green', linestyle='--', alpha=0.8, linewidth=0.6)
    
    # 빨간색 수평선/수직선 (top-1 cost)
    if top1_cost_val is not None:
        plt.axhline(y=top1_cost_val, color='red', linestyle='--', alpha=0.8, linewidth=0.6, label=f'Top-1 cost: {top1_cost_val:.2f}')
        plt.axvline(x=t[top1_index], color='red', linestyle='--', alpha=0.8, linewidth=0.6)
    
    if show:
        plt.colorbar(sc, label="monotonicity error (blue=good, red=bad)")
        plt.xlabel("t (Progress along Learned Curved Trajectory)")
        plt.ylabel("cost (aligned)")
        plt.title(f"{title_prefix}Monotonicity error colored (Curved Trajectory)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Top-1 cost value (aligned): {top1_cost_val:.4f}")
    # top-1이 t축 기준으로 맨 끝에서 몇 번째인지
    top1_t_rank = np.argsort(np.argsort(t))[top1_index]
    print(f"Top-1 curve-t rank: {len(t)-top1_t_rank}")
    
    print(f"Max curve-t cost value (aligned): {max_t_cost_val:.4f}")
    print(f"Max t value: {t.max():.4f}")

    print(f"curve-t Top-10 mean : {y_plot[np.argsort(t)[-10:]].mean():.4f}")
    print(f"spearman_rho : {rho:.4f}")
    print(f"kendall_tau : {tau:.4f}")

    if not show:
        plt.close()
        results = {
            "true max rank": len(t)-top1_t_rank,
            "true max cost": round(top1_cost_val, 4),
            "max cost": round(max_t_cost_val, 4),
            "top-10 mean": round(y_plot[np.argsort(t)[-10:]].mean(), 4),
            "spearman_rho": rho,
            "kendall_tau": tau,
        }
        return results