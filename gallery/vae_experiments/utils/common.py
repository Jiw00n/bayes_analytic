import torch
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
