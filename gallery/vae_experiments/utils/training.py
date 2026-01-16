import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import itertools
import torch
import pandas as pd
from tvm import auto_scheduler
from utils.model import VAE_feature_head
from utils.common import seed_everything, pair_accuracy, recall_at_k
from tqdm import tqdm


def vae_feature_loss(x_recon, x, mu, logvar, feature_pred, feature, alpha_recon=0, alpha_feature=0, beta=1.0):
    """
    x, x_recon: (B, input_dim)
    mu, logvar: (B, latent_dim)

    beta: KL 가중치 (β-VAE 스타일로 조절)
    """
    # reconstruction loss: MSE
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # 
    # recon_loss = L3Loss()(x_recon, x)

    feature_loss = F.mse_loss(feature_pred, feature, reduction="mean") if feature_pred is not None else 0.0

    # KL divergence: D_KL(q(z|x) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = alpha_recon * recon_loss + beta * kl + alpha_feature * feature_loss
    return loss, recon_loss, kl, feature_loss



def vae_train(input_dim, train_loader, val_loader, device, train_seed=42):

    latent_dim = 64
    hidden_dim = 256


    hyperparameter = {
        'beta': [0.01],
        'alpha_recon': [1.0],
        'alpha_feature': [0.0],
        'latent_dim': [64],
        'lr': [1e-3],
    }

    cnt = 0
    epochs = 500


    for vals in itertools.product(*hyperparameter.values()):
        (beta, alpha_recon, alpha_feature, latent_dim, lr) = vals
        cnt += 1
        print("=============================================")
        print(f"Experiment {cnt}/{len(list(itertools.product(*hyperparameter.values())))}")
        print(f"beta={beta}, alpha_recon={alpha_recon}, alpha_feature={alpha_feature},\nepochs={epochs}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, lr={lr}")

        seed_everything(train_seed)

        vae = VAE_feature_head(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

        # early stopping
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0

        for epoch in range(1, epochs+1):
            vae.train()
            for x_batch in train_loader:
                if len(x_batch) == 2:
                    x_batch, feature_batch = x_batch
                    feature_batch = feature_batch.to(device)
                else:
                    feature_batch = None
                x_batch = x_batch.to(device)  # (N, D)
                
                

                x_recon, mu, logvar, z, feature_pred = vae(x_batch, use_mean=False)

                loss, recon_loss, kl, feature_loss = vae_feature_loss(x_recon, x_batch, mu, logvar, feature_pred, feature_batch, alpha_recon=alpha_recon, alpha_feature=alpha_feature, beta=beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if epoch % 50 == 0:
                vae.eval()
                for x_batch in val_loader:
                    if len(x_batch) == 2:
                        x_batch, feature_batch = x_batch
                        feature_batch = feature_batch.to(device)
                    else:
                        feature_batch = None
                    x_batch = x_batch.to(device)
                    if feature_batch is not None:
                        feature_batch = feature_batch.to(device)
                    x_recon, mu, logvar, z, feature_pred = vae(x_batch, use_mean=True)
                    val_loss, val_recon_loss, val_kl, val_feature_loss = vae_feature_loss(x_recon, x_batch, mu, logvar, feature_pred, feature_batch, alpha_recon=alpha_recon, alpha_feature=alpha_feature, beta=beta)
                    val_recon_r2 = r2_score(x_batch.detach().cpu().numpy(), x_recon.detach().cpu().numpy())
                    if feature_batch is not None:
                        val_feature_r2 = r2_score(feature_batch.detach().cpu().numpy(), feature_pred.detach().cpu().numpy())
                    else:
                        val_feature_r2 = None

                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                    best_vae = vae
                    best_recon_r2 = val_recon_r2
                    best_feature_r2 = val_feature_r2
                else:
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                print(f"epoch {epoch}")
                print(f"loss={loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl.item():.4f}")
                print(f"val loss={val_loss.item():.4f}, val recon={val_recon_loss.item():.4f}, val kl={val_kl.item():.4f}")
                print(f"Recon R2 : {val_recon_r2}, Feature R2 : {val_feature_r2}")

    print(f"Final VAE : Recon R2 : {best_recon_r2}, Feature R2 : {best_feature_r2}")
    return best_vae




def reg_loss_fn(cost_pred, cost_true, loss_type='mse'):
    """
    기본 회귀 손실 (MSE 또는 MAE)
    """
    if loss_type == 'mse':
        return F.mse_loss(cost_pred, cost_true)
    else:  # mae
        return F.l1_loss(cost_pred, cost_true)


def pair_loss_fn(cost_pred, cost_true, margin=0.1):
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


def smooth_loss_fn(model, z, noise_std=0.1):
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


def kld_loss_fn(mean, logvar):
    """
    KL Divergence: q(z|x) || N(0, I)
    """
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return kld

def feature_loss_fn(use_feature, feature_pred, feature_true, coef=0.1):
    """
    Feature 예측 손실 (MSE)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_feature:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(feature_pred, feature_true) * coef


def compute_total_loss(model, cost_pred, mean, logvar, z, labels, feature, config, return_components=True):
    """
    Total loss 계산 (Segment 기반 데이터용).
    total_loss = reg_loss + λ_pair * pair_loss + γ * smooth_loss + β * kld_loss
    """
    
    # Individual losses
    reg = reg_loss_fn(cost_pred, labels, loss_type=config.get('loss_type', 'mse'))
    pair = pair_loss_fn(cost_pred.view(-1), labels.view(-1), margin=config.get('margin', 0.1))
    smooth = smooth_loss_fn(model, z, noise_std=config.get('noise_std', 0.1))
    kld = kld_loss_fn(mean, logvar)
    feature_loss = feature_loss_fn(model.use_feature, None, feature, coef=0)
    
    # Weighted sum
    total = config['lambda_reg'] * reg + config['lambda_pair'] * pair + config['gamma'] * smooth + config['beta'] * kld + feature_loss
    
    if return_components:
        return total, {
            'reg_loss': reg.item(),
            'pair_loss': pair.item(),
            'smooth_loss': smooth.item(),
            'kld_loss': kld.item(),
            'feature_loss': feature_loss.item(),
        }
    return total



def train_regression(vae_cost_model, optimizer, train_loader, config, device):

    print("Train size :", len(train_loader.dataset))
   

    for epoch in tqdm(range(1, config['epochs']+1), desc="Training Epochs"):
        vae_cost_model.train()

        epoch_reg = 0.0
        epoch_pair = 0.0
        epoch_kl = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Train", leave=False)
        for x_batch, labels in pbar:
            x_batch = x_batch.to(device)
            labels = labels.to(device).squeeze(-1)
            
        
            cost_pred, mean, logvar, z = vae_cost_model(x_batch, use_mean=True)

            train_loss, train_components = compute_total_loss(vae_cost_model, 
                                                    cost_pred, mean, logvar, z, labels, None, config)

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_cost_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_reg += train_components['reg_loss'].item()
            epoch_pair += train_components['pair_loss'].item()
            epoch_kl += train_components['kld_loss'].item()
            num_batches += 1

            pbar.set_postfix({
                "reg": f"{train_components['reg_loss'].item():.4f}",
                "rank": f"{train_components['pair_loss'].item():.4f}",
                "kl": f"{train_components['kld_loss'].item():.4f}"
            })

        if epoch % config['epochs'] == 0:
            print(f"[Train epoch {epoch}] : reg={train_components['reg_loss']: .4f} rank={train_components['pair_loss']: .4f} kl={train_components['kld_loss']: .4f}")
        


    return vae_cost_model



def validate_regression(vae_cost_model, input_data_scaled, costs, selected_indices, used_indices, device, config, use_cumul=True):
    """
    
    """

    val_input = input_data_scaled[selected_indices]
    val_costs = costs[selected_indices]
    cumul_input = input_data_scaled[list(used_indices)]
    cumul_costs = costs[list(used_indices)]

    val_input = torch.from_numpy(val_input).float()
    val_costs = torch.from_numpy(val_costs).float()
    

    vae_cost_model.eval()
    with torch.no_grad():
        val_costs = val_costs.to(device).squeeze(-1)
        cost_pred, mean, logvar, z = vae_cost_model(val_input.to(device), use_mean=True)

        val_loss, val_components = compute_total_loss(vae_cost_model, cost_pred, mean, logvar, z, val_costs, None, config)
        val_reg_r2 = r2_score(torch.cat(cost_pred).detach().cpu().numpy(), torch.cat(val_costs).detach().cpu().numpy())
        val_reg_r2 = round(val_reg_r2, 4)
        val_rank_r2 = pair_accuracy(cost_pred, val_costs)
        val_rank_r2 = round(val_rank_r2, 4)
        print(f"Val total={val_loss} reg={val_components['reg_loss']: .4f} rank={val_components['pair_loss']: .4f} kl={val_components['kld_loss']: .4f}")
        print(f"Regression R2 : {val_reg_r2:.4f}, Rank R2 : {val_rank_r2:.4f}")
        
        cumul_rank_r2 = None
        if use_cumul:
            cumul_input = torch.from_numpy(cumul_input).float()
            cumul_costs = torch.from_numpy(cumul_costs).float()
            cumul_preds = vae_cost_model(cumul_input.to(device), use_mean=True)[0].detach().cpu().numpy()
        
            cumul_rank_r2 = pair_accuracy(cumul_preds, val_costs)
            cumul_rank_r2 = round(cumul_rank_r2, 4)
            print(f"Rank R2 : {cumul_rank_r2:.4f}")

        return val_reg_r2, val_rank_r2, cumul_rank_r2