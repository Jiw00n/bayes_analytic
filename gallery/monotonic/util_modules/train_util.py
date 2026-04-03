import random
import torch
import numpy as np



def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_split_exclude_max(input_data_scaled, costs, train_size, seed):


    rng = np.random.default_rng(seed=seed)

    N = len(costs)

    # 1. costs 최대값 인덱스 (여러 개면 전부)
    max_idx = np.where(costs == np.max(costs))[0]

    # 2. 최대값 제외한 나머지 인덱스
    rest_idx = np.setdiff1d(np.arange(N), max_idx)

    # 3. 셔플
    rng.shuffle(rest_idx)

    # 4. train / val 분리
    train_idx = rest_idx[:train_size]
    val_idx = np.concatenate([rest_idx[train_size:], max_idx])

    # 5. 데이터 분리
    X_train = input_data_scaled[train_idx]
    y_train = costs[train_idx]

    X_val = input_data_scaled[val_idx]
    y_val = costs[val_idx]


    return X_train, X_val, y_train, y_val