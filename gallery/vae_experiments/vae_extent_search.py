import pandas as pd
import datetime
import itertools
import time
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import FeatureDataset, make_vae_reg_dataloaders
from utils.select import select_programs, random_select_indices, select_init_latent_diversity
from utils.common import seed_everything
from utils.model import make_vae_reg_model
from utils.training import vae_train, train_regression, validate_regression
from utils.extent import gen_program_pool
from utils.util_manager import get_network, get_tasks
import tvm
from tvm import auto_scheduler
import pickle
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = tvm.target.Target("cuda")


train_seed = 2023
seed_everything(train_seed)



network_name = "resnet_50"

mod, params, input_shape, output_shape = get_network(network_name, 1, "NHWC", dtype="float32")
tasks, task_weights = get_tasks(mod, params, network_name, input_shape, TARGET, get_pkl=True)
nn_conv2d_add_nn_relu = [0, 3, 7, 10, 18, 20, 24, 26]
breakpoint()


size = 4000



tic = time.time()
records_sch = []
records_all = []
for task in [tasks[i] for i in nn_conv2d_add_nn_relu]:
    records = gen_program_pool(task, size, evo_population=2560, min_population=100, seed=train_seed)
    records_all.append(records["all"])
    records_sch.append(records["schedules"])

breakpoint()
############################### vae training ###############################


input_data = np.log1p(np.array(records["all"], dtype=np.float32))

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)
input_dim = input_data_scaled.shape[-1]

X_train, X_val = train_test_split(
    input_data_scaled,  test_size=0.2, random_state=train_seed
)


# feature 없음
train_dataset = FeatureDataset(X_train)
val_dataset   = FeatureDataset(X_val)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)



vae = vae_train(input_dim, train_loader, val_loader, device, train_seed=train_seed)



breakpoint()

############################### active learning ###############################



# 데이터셋 길이만큼의 인덱스 numpy 배열 생성
all_indices = np.arange(len(input_data_scaled))
# costs = np.array(records["costs"], dtype=np.float32)

# real_optimum_index = np.argmax(costs)



top_k = 1

sampling_hyper = {
    "measure_size": [64],
    "weight" : [
            # (1.0, 0.0, 0.0),
            # (0.7, 0.0, 0.3),
            # (0.7, 0.3, 0.0),
            # (0.6, 0.1, 0.3),
            # (0.3, 0.4, 0.3),
            (0.4, 0.3, 0.3),
            # (0.3, 0.3, 0.4),
            # (0.5, 0.2, 0.3),
            ],
    "uncertainty_topk": [64],
    # "weight" : f_weights,
    "grad_num": [4],
    "rand_num": [0],
    
    "T_mc": [20],
    # "seed" : range(2000, 2010),
    # "seed" : [2023,2025],
}

random_indices_list = []
all_results = []
costs = np.zeros(len(input_data_scaled), dtype=np.float32)

cnt = 0

filename = f"result_ansor/vae_extent_{tic}.csv"

for params in itertools.product(*sampling_hyper.values()):

    cnt += 1
    print(f"########## 실험 {cnt}/{len(list(itertools.product(*sampling_hyper.values())))} ##########")

    # used_indices : 이미 측정된 인덱스 집합. train_indices와 동일
    # remaining_indices : 아직 측정되지 않은 인덱스 집합. val_indices와 동일
    used_indices = set()
    remaining_indices = set(all_indices)
    
    measure_size, weight, uncertainty_topk, grad_num, rand_num, T_mc, sampling_seed = params
    w_cost, w_unc, w_div = weight
    print(f"weights: {weight}")
    print(f"measure_size: {measure_size}, T_mc: {T_mc}, sampling_seed: {sampling_seed}")

    sampling_rng = np.random.default_rng(sampling_seed)

    hyperparameter = {

        'lambda_reg' : [0.01],
        'lambda_pair': [3.0],
        'margin_scale': [0.3],
        'gamma': [0.01],
        'beta': [0.01],
        'noise_std': [0.001],

        'encoder_lr': [1e-4],
        'feature_predictor_lr': [0],
        'cost_predictor_lr': [1e-2],
        'epochs': [1000],
        
    }

    input_dim = input_data_scaled.shape[-1]
    latent_dim = 64
    hidden_dim = 256



    
    random_indices, remaining_indices = random_select_indices(remaining_indices, select_size=sampling_hyper["measure_size"][0], rng=sampling_rng)
    # random_indices, remaining_indices = select_init_latent_diversity(vae, input_data_scaled, remaining_indices, select_num=sampling_hyper["measure_size"][0])
    print(f"초기 랜덤 선택 샘플 인덱스: {np.sort(random_indices)}")
    used_indices.update(random_indices)
    random_indices_list.append(random_indices)

    reg_history = []
    rank_history = []
    cumul_rank_history = []

    for phase in range(1, len(input_data_scaled) // measure_size + 1):

        print(f"=============== 측정 Phase {phase} ================")


        # DataLoader 갱신
        seed_everything(train_seed)
        train_loader, val_loader, y_mean, y_std = make_vae_reg_dataloaders(input_data_scaled, costs, used_indices, remaining_indices)
        vae_cost_model, optimizer, config = make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std, device, verbose=False)
        
        
        seed_everything(train_seed)
        vae_cost_model, topk_recall_signal, val_reg_r2, val_rank_r2 = train_regression(vae_cost_model, optimizer, train_loader, config, device)


        # 다음 측정할 샘플 선택
        selected_indices, remaining_indices = select_programs(
            model=vae_cost_model,
            input_data_scaled=input_data_scaled,
            remaining_indices=remaining_indices,
            used_indices=used_indices,
            num_select=measure_size,
            T_mc=T_mc,
            w_cost=weight[0],
            w_unc=weight[1],
            w_div=weight[2],
            # w_cost=0.3,
            # w_unc=0.35,
            # w_div=0.35,
            uncertainty_topk=uncertainty_topk,
            grad_num=grad_num,
            rand_num=rand_num,
            device=device,
            rng=sampling_rng,
            
            topk_factor=5
        )
        # w_cost += 0.03
        # w_unc -= 0.02
        # w_div -= 0.01

        # selected_indices: numpy 배열
        used_indices.update(selected_indices.tolist())

        # measure

        costs[selected_indices]



        val_reg_r2, val_rank_r2, cumul_rank_r2 = validate_regression(vae_cost_model, input_data_scaled, costs, selected_indices, used_indices, device, config, use_cumul=True)
        reg_history.append(val_reg_r2)
        rank_history.append(val_rank_r2)
        cumul_rank_history.append(cumul_rank_r2)

        used_time = time.time() - tic
        print(f"총 측정 시간: {used_time:.2f} 초")
        print("=============================================")
        all_results.append({
            "measure_size": measure_size,
            "weights": weight,
            "uncertainty_topk": uncertainty_topk,
            "grad_num": grad_num,
            "rand_num": rand_num,
            "phase" : phase,
            "used_time": round(used_time, 2),
            "train_size" : len(used_indices)-measure_size,
            "val_reg_r2": reg_history,
            "val_rank_r2": rank_history,
            "val_cumul_rank_r2": cumul_rank_history,
            "sampling_seed": sampling_seed,
        })

        df_results = pd.DataFrame(all_results)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_results.to_csv(filename, index=False)
        


group_cols = [
    "measure_size",
    "weights",
    "uncertainty_topk",
    "grad_num",
    "rand_num",
]

agg_dict = {
    "phase": "mean",
    "train_size": "mean",
    "used_time": "mean",
    "val_reg_r2": "first",
    "val_rank_r2": "first",
}

df_avg = (
    df_results
    .groupby(group_cols, as_index=False)
    .agg(agg_dict)
)
print(df_avg)