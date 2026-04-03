import torch
import numpy as np
import random


def pair_accuracy(cost_pred, labels, rng=np.random.default_rng(42)):
    """
    cost_pred, labels: (B,) 텐서
    """
    n_samples = min(2000, len(cost_pred))
    sample_indices = rng.choice(len(cost_pred), n_samples, replace=False)

    correct = 0
    total = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            idx_i = sample_indices[i]
            idx_j = sample_indices[j]
            pred_diff = cost_pred[idx_i] - cost_pred[idx_j]
            true_diff = labels[idx_i] - labels[idx_j]
            if (pred_diff * true_diff) > 0:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def recall_at_k(pred, labels, k=1):
    true_best_idx = torch.argmax(labels)
    topk_pred_idx = torch.topk(pred, k=k, largest=True).indices

    return int((topk_pred_idx == true_best_idx).any())



import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


# X_train으로부터 cost 방향을 추출하고,
# 그 방향에 대한 global monotonicity와 local smoothness를 평가
def analyze_feature_structure(X_train, y_train, X_val, y_val, k=5):
    
    # -------------------------
    # 1. cost direction 추출
    # -------------------------
    def fit_cost_direction(X_train, y_train, alpha=1.0):
        """
        feature -> cost 선형 방향 추출
        """
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        w = reg.coef_
        w = w / (np.linalg.norm(w) + 1e-12)
        return w


    def project(X, w):
        return X @ w


    # -------------------------
    # 2. 전역 정렬성 평가
    # -------------------------
    def eval_global_ordering(proj, y):
        """
        projection vs cost monotonicity
        """
        rho, _ = spearmanr(proj, y)
        return rho


    # -------------------------
    # 3. local smoothness 평가
    # -------------------------
    def eval_local_smoothness(X, y, k=10):
        """
        kNN 이웃들의 cost 분산
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]

        local_vars = []
        for i in range(len(y)):
            local_vars.append(np.var(y[idx[i]]))

        return {
            "mean_var": float(np.mean(local_vars)),
            "median_var": float(np.median(local_vars))
        }
    
    # X_train으로부터 cost 방향을 추출하고,
    # 그 방향에 대한 global monotonicity와 local smoothness를 평가
    w = fit_cost_direction(X_train, y_train)

    proj_train = project(X_train, w)
    proj_val   = project(X_val, w)

    train_local = eval_local_smoothness(X_train, y_train, k)
    val_local   = eval_local_smoothness(X_val, y_val, k)

    result = {
        "train_spearman": eval_global_ordering(proj_train, y_train),
        "val_spearman":   eval_global_ordering(proj_val, y_val),
        "train_local_mean_var":    train_local["mean_var"],
        "train_local_median_var":  train_local["median_var"],
        "val_local_mean_var":      val_local["mean_var"],
        "val_local_median_var":    val_local["median_var"],
    }
    return result

