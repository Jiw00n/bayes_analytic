import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tvm import auto_scheduler


class FeatureRegressionDataset(Dataset):
    def __init__(self, X, y, feature=None):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X
        self.y = torch.from_numpy(y).float()
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

        self.feature = feature
        if feature is not None:
            if isinstance(feature, np.ndarray):
                self.feature = torch.from_numpy(feature).float()
            else:
                self.feature = feature
            
            if self.feature.ndim == 1:
                self.feature = self.feature.unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.feature is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.feature[idx]


class FeatureDataset(Dataset):
    def __init__(self, X, feature=None):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X
        
        if isinstance(feature, np.ndarray):
            self.feature = torch.from_numpy(feature).float()
        else:
            self.feature = feature
        # feature shape이 (N,)이면 (N,1)로 바꿔주는 게 편할 때가 많음
        if self.feature is not None and self.feature.ndim == 1:
            self.feature = self.feature.unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.feature is None:
            return self.X[idx]
        return self.X[idx], self.feature[idx]
    



def make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices):

    train_indices = np.array(list(used_indices), dtype=np.int64)
    val_indices = np.array(list(remaining_indices), dtype=np.int64)

    X_train = input_data_scaled[train_indices]
    X_val = input_data_scaled[val_indices]

    if costs.shape[0] == input_data_scaled.shape[0]:
        y_train = costs[train_indices]
        y_val = costs[val_indices]

        train_dataset = FeatureRegressionDataset(X_train, y_train)
        val_dataset   = FeatureRegressionDataset(X_val,   y_val)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8  # 0 나누기 방지용 작은 값 추가
        print(f"y_train mean: {y_mean}, std: {y_std}")

        return train_loader, val_loader, y_mean, y_std
    else:
        y_train = costs[train_indices]
        y_val = costs[val_indices]

        train_dataset = FeatureRegressionDataset(X_train, y_train)
        val_dataset   = FeatureRegressionDataset(X_val,   y_val)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8  # 0 나누기 방지용 작은 값 추가
        print(f"y_train mean: {y_mean}, std: {y_std}")

        return train_loader, val_loader, y_mean, y_std



    

    
    


def make_xgb_datasets(inputs, results):
    f_inputs = []
    f_results = []
    r_costs = []
    for inp, res in zip(inputs, results):
        cost = np.mean([c.value for c in res.costs])
        if cost < 1e10:
            f_inputs.append(inp)
            f_results.append(res)
            r_costs.append(cost)
    r_costs = np.array(r_costs, dtype=np.float32)
    
    dataset = auto_scheduler.dataset.Dataset()
    dataset.update_from_measure_pairs(f_inputs, f_results)
    return dataset


def split_xgb_datasets(dataset, train_indices, test_indices):

    raw_features = list(dataset.features.values())[0]
    raw_throughputs = list(dataset.throughputs.values())[0]

    
    train_set, test_set = dataset.random_split_within_task(train_set_ratio=0, 
                                                        train_idxs=train_indices.tolist(), 
                                                        test_idxs=test_indices.tolist())
    return train_set, test_set, raw_throughputs