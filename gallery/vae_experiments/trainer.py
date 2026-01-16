import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
import random
from models.regression import VAECostPredictor
from util_manager import SEED, seed_everything
from models.vae import SegmentVAE
from tqdm import tqdm



seed_everything(SEED)

# 하이퍼파라미터 탐색을 위한 VAE 학습 함수
# KL/latent_dim이 0.05~0.2 범위에 있도록 우선 조정
from itertools import product




class VAE_Trainer:
    def __init__(self, train_loader, val_loader, input_dim=164, hidden_dim=256, latent_dim=128, dropout=0.1, device='cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.device = device


    def train_vae(self, epochs=200, lr=1e-3, beta=0.1, patience=30, verbose=True):
        """단일 설정으로 VAE 학습"""

        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        latent_dim = self.latent_dim
        dropout = self.dropout
        
        model = SegmentVAE(input_dim, hidden_dim, latent_dim, dropout).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
        
        best_recon_r2 = -float('inf')
        patience_counter = 0
        best_state = None
        history = {'recon_r2': [], 'kl_per_dim': [], 'recon_loss': []}
        
        for epoch in range(epochs):
            model.train()
            epoch_kl = 0.0
            epoch_recon = 0.0
            n_batches = 0
            
            for segment_sizes_batch, features_batch, _ in self.train_loader:
                features_batch = features_batch.to(self.device)
                segment_sizes_batch = segment_sizes_batch.to(self.device)
                mean, logvar, z, recon, segment_sum_vec = model(segment_sizes_batch, features_batch)
                loss, recon_loss, kld_loss = self.vae_loss(recon, segment_sum_vec, mean, logvar, beta)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # KL per dimension 계산 (평균)
                batch_kl_per_dim = kld_loss.item() / (mean.shape[0] * latent_dim)
                epoch_kl += batch_kl_per_dim
                epoch_recon += recon_loss.item() / mean.shape[0]
                n_batches += 1
            
            scheduler.step()
            avg_kl_per_dim = epoch_kl / n_batches
            avg_recon = epoch_recon / n_batches
            
            # Validation
            model.eval()
            val_recons, val_originals = [], []
            val_kl_per_dim = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for segment_sizes_batch, features_batch, _ in self.val_loader:
                    features_batch = features_batch.to(self.device)
                    segment_sizes_batch = segment_sizes_batch.to(self.device)
                    mean, logvar, z, recon, segment_sum_vec = model(segment_sizes_batch, features_batch, use_mean=True)
                    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                    val_kl_per_dim += kld.item() / (mean.shape[0] * latent_dim)
                    val_batches += 1
                    val_recons.append(recon.cpu().numpy())
                    val_originals.append(segment_sum_vec.cpu().numpy())
            
            val_kl_per_dim /= val_batches
            recon_r2 = r2_score(
                np.concatenate(val_originals).flatten(),
                np.concatenate(val_recons).flatten()
            )
            
            history['recon_r2'].append(recon_r2)
            history['kl_per_dim'].append(val_kl_per_dim)
            history['recon_loss'].append(avg_recon)
            
            if recon_r2 > best_recon_r2:
                best_recon_r2 = recon_r2
                patience_counter = 0
                best_state = model.state_dict().copy()
                best_kl_per_dim = val_kl_per_dim
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: R²={recon_r2:.4f}, KL/dim={val_kl_per_dim:.4f}")
            
            if patience_counter >= patience:
                break
        
        model.load_state_dict(best_state)
        return model, best_recon_r2, best_kl_per_dim, history




    def hyperparameter_search(self, epochs=300, patience=30, target_kl_range=(0.05, 0.4)):
        """
        하이퍼파라미터 탐색:
        - KL/latent_dim이 0.05~0.2 범위에 있도록 우선 조정
        - 범위를 벗어나면 페널티 부여
        """
        
        # 하이퍼파라미터 그리드

        configs = {
            "hidden_dim": [256, 512],
            "latent_dim": [64, 128],
            "beta": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "lr": [1e-3, 5e-4, 2e-4],
        }

        all_results = []
        best_score = -float('inf')
        best_config = None
        best_model = None
        
        config_idx = 0
        configs = [
            {'hidden_dim': 256, 'latent_dim': 64,  'beta': 1e-04, 'lr': 1e-03},
            # {'hidden_dim': 256, 'latent_dim': 64,  'beta': 1e-04, 'lr': 2e-04},
            # {'hidden_dim': 256, 'latent_dim': 64,  'beta': 2e-04, 'lr': 1e-03},

            # {'hidden_dim': 256, 'latent_dim': 128, 'beta': 5e-05, 'lr': 1e-03},
            # {'hidden_dim': 256, 'latent_dim': 128, 'beta': 5e-05, 'lr': 5e-04},
            # {'hidden_dim': 256, 'latent_dim': 128, 'beta': 5e-05, 'lr': 2e-04},
            # {'hidden_dim': 256, 'latent_dim': 128, 'beta': 1e-04, 'lr': 2e-04},
            # {'hidden_dim': 256, 'latent_dim': 128, 'beta': 1e-03, 'lr': 2e-04},
        ]





        if isinstance(configs, dict):
            keys = list(configs.keys())
            values_product = list(product(*[configs[k] for k in keys]))
            config_list = [dict(zip(keys, vals)) for vals in values_product]
        elif isinstance(configs, list):
            # 이미 [{"hidden_dim":..., "latent_dim":...}, ...] 형태라고 가정
            config_list = configs
        else:
            raise ValueError("configs는 dict (각 값이 리스트) 이거나 dict의 리스트여야 합니다.")

        total_configs = len(config_list)


        for config in config_list:
            hidden_dim = config['hidden_dim']
            latent_dim = config['latent_dim']
            beta = config['beta']
            lr = config['lr']
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}] hidden={hidden_dim}, latent={latent_dim}, β={beta:.0e}, lr={lr:.0e}")
            
            try:
                model, recon_r2, kl_per_dim, history = self.train_vae(
                    beta=beta,
                    lr=lr,
                    epochs=epochs,
                    patience=patience,
                    verbose=False
                )
                
                # KL/dim 범위 체크
                kl_min, kl_max = target_kl_range
                in_kl_range = kl_min <= kl_per_dim <= kl_max
                
                # 스코어 계산: R² 기준, KL 범위 벗어나면 페널티
                if in_kl_range:
                    score = recon_r2
                    kl_status = "✓"
                else:
                    # 범위를 벗어난 정도에 따라 페널티
                    if kl_per_dim < kl_min:
                        penalty = (kl_min - kl_per_dim) / kl_min
                    else:
                        penalty = (kl_per_dim - kl_max) / kl_max
                    score = recon_r2 - penalty * 0.1  # 최대 10% 페널티
                    kl_status = "✗"
                
                result = {
                    **config,
                    'recon_r2': recon_r2,
                    'kl_per_dim': kl_per_dim,
                    'in_kl_range': in_kl_range,
                    'score': score,
                    'model': model,
                    'history': history
                }
                all_results.append(result)
                
                print(f"  → R²={recon_r2:.4f}, KL/dim={kl_per_dim:.4f} {kl_status}, Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    best_model = model
                    best_result = result
                    print(f"  ★ 새로운 최고!")
                
            except Exception as e:
                print(f"  오류: {e}")
                continue
        
        # 결과 정렬 및 출력
        print("\n" + "=" * 70)
        print("탐색 결과 (Score 기준 상위 10개)")
        print("=" * 70)
        
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
        print(f"{'Rank':<5} {'Hidden':<8} {'Latent':<8} {'β':<10} {'LR':<10} {'R²':<8} {'KL/dim':<10} {'KL범위':<8} {'Score':<8}")
        print("-" * 85)
        
        for i, res in enumerate(sorted_results[:10]):
            kl_status = "✓" if res['in_kl_range'] else "✗"
            print(f"{i+1:<5} {res['hidden_dim']:<8} {res['latent_dim']:<8} {res['beta']:<10.0e} {res['lr']:<10.0e} {res['recon_r2']:<8.4f} {res['kl_per_dim']:<10.4f} {kl_status:<8} {res['score']:<8.4f}")
        
        return best_model, best_config


    def vae_loss(self, recon, original, mean, logvar, beta):
        """
        VAE Loss = Reconstruction Loss + β * KL Divergence
        - Reconstruction: MSE between original and reconstructed segment sum 벡터
        - KLD: D_KL(q(z|x) || p(z)), where p(z) = N(0, I)
        """
        recon_loss = F.mse_loss(recon, original, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss





######## regression trainer ########



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import copy
import torch.nn.functional as F



class Regression_Trainer:
    def __init__(self, vae_model, config, device='cuda'):
        """
        Regression Trainer
        
        Args:
            vae_model: 학습된 VAE 모델 정보 (dict 형태)
            input_dim: 입력 feature 차원
            device: 학습 디바이스
        """
        self.vae_model = vae_model
        self.config = config
        self.device = device
        self.model = VAECostPredictor(
            input_dim=164,
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            predictor_hidden=256,
            predictor_layers=3,
            dropout=0.08
        ).to(self.device)
        self.model.load_pretrained_encoder(vae_model.state_dict())

    def train_regression(self, train_feature_list, train_segment_sizes, train_labels,
                        fea_norm_vec, config):
        """
        회귀 모델 학습
        
        Args:
            train_feature_list: 학습 feature 리스트
            train_segment_sizes: 각 샘플의 segment 크기
            train_labels: 학습 레이블 (cost)
            fea_norm_vec: Feature 정규화 벡터
            config: 학습 설정 (lr, lambda_pair, gamma, beta 등)
            phases: 학습 phase 수
        """
        print("=" * 70)
        print("🎯 Regression 모델 학습 시작")
        print("=" * 70)

        # 모델 생성
        

        print(f"   VAE Config: hidden_dim={self.config['hidden_dim']}, "
              f"latent_dim={self.config['latent_dim']}")

        # 파라미터 수 출력
        enc_params = sum(p.numel() for p in self.model.get_encoder_params())
        cost_params = sum(p.numel() for p in self.model.get_predictor_params())
        print(f"\n모델 파라미터 수:")
        print(f"   Encoder: {enc_params:,}")
        print(f"   Cost Predictor: {cost_params:,}")
        print(f"   Total: {enc_params + cost_params:,}\n")

        # Phase 학습 실행
        self.model, history, best_loss = self.train_phase(
            self.model, train_feature_list, train_segment_sizes, train_labels,
            fea_norm_vec, self.device, config,
            samples_per_phase=len(train_labels),
            mini_epochs=config.get('mini_epochs', 100),
        )
        
        return history, best_loss

    def train_phase(self, model, train_feature_list, train_segment_sizes, train_labels,
                                fea_norm_vec, device, config,
                                samples_per_phase=64, mini_epochs=10):
        """
        Phase 최적화 학습 (모든 데이터 학습)
        
        Args:
            model: 학습할 모델
            train_feature_list: 학습 feature 리스트
            train_segment_sizes: 각 샘플의 segment 크기
            train_labels: 학습 레이블 (cost)
            fea_norm_vec: Feature 정규화 벡터
            device: 학습 디바이스
            config: 학습 설정
            samples_per_phase: Phase당 샘플 수
            mini_epochs: 각 Phase 내 반복 횟수
            phases: 총 Phase 수
        """
        print("=" * 70)
        print(f"🎯 학습 시작")
        print(f"   Total samples: {len(train_labels)}개")
        print(f"   Samples per phase: {samples_per_phase}")
        print(f"   Mini-epochs per phase: {mini_epochs}")
        print("=" * 70)
        
        # 인덱스 준비
        all_indices = np.arange(len(train_labels))
        # np.random.shuffle(all_indices)  # 필요시 셔플
        
        best_loss = float('inf')
        best_state = None
        history = []
        
        # Optimizer 설정
        optimizer = torch.optim.AdamW([
            {'params': model.get_encoder_params(), 'lr': config['encoder_lr']},
            {'params': model.get_predictor_params(), 'lr': config['predictor_lr']}
        ], weight_decay=1e-5)

        total_samples_used = 0
        
        for phase in range(1):
            
            # Phase 샘플 선택
            start_idx = phase * samples_per_phase
            end_idx = min(start_idx + samples_per_phase, len(all_indices))
            phase_indices = all_indices[start_idx:end_idx].tolist()
            total_samples_used += len(phase_indices)
            
            
            # Phase 데이터 준비
            phase_features = [train_feature_list[i] for i in phase_indices]
            phase_segment_sizes = train_segment_sizes[phase_indices]
            phase_labels = train_labels[phase_indices]
            
            # Flatten features
            flatten_features = np.concatenate(phase_features, axis=0).astype(np.float32)
            
            # Tensor 변환
            segment_sizes_tensor = torch.tensor(phase_segment_sizes, dtype=torch.int32).to(device)
            features_tensor = torch.tensor(flatten_features, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(phase_labels, dtype=torch.float32).to(device)
            
            # 정규화
            if fea_norm_vec is not None:
                features_tensor = features_tensor / fea_norm_vec.to(device)
            
            # Mini-epochs 학습
            for epoch in range(mini_epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward
                total_loss, components = self.compute_total_loss(
                    model, segment_sizes_tensor, features_tensor, labels_tensor, 
                    config, return_components=True
                )
                
                # Backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 로깅 (5 epoch마다)
                if (epoch + 1) % 5 == 0 or epoch == mini_epochs - 1:
                    print(f"   [Epoch {epoch+1}/{mini_epochs}] "
                          f"Loss: {components['total_loss']:.4f} "
                          f"(Reg: {components['reg_loss']:.4f}, "
                          f"Pair: {components['pair_loss']:.4f}, "
                          f"Smooth: {components['smooth_loss']:.4f}, "
                          f"KLD: {components['kld_loss']:.4f})")
                    
                    history.append({
                        'phase': phase + 1,
                        'epoch': epoch + 1,
                        **components
                    })
                    
                    # Best model 저장 (loss 기준)
                    if components['total_loss'] < best_loss:
                        best_loss = components['total_loss']
                        best_state = copy.deepcopy(model.state_dict())
        
        # Best model 복원
        if best_state is not None:
            model.load_state_dict(best_state)
        
        print(f"{'='*70}")
        print(f"   Best Loss: {best_loss:.4f}")
        print(f"   사용된 샘플: {total_samples_used}개 / 전체 {len(train_labels)}개")
        
        return model, history, best_loss


    def reg_loss_fn(self, cost_pred, cost_true, loss_type='mse'):
        """
        기본 회귀 손실 (MSE 또는 MAE)
        """
        if loss_type == 'mse':
            return F.mse_loss(cost_pred, cost_true)
        else:  # mae
            return F.l1_loss(cost_pred, cost_true)


    def pair_loss_fn(self,cost_pred, cost_true, margin=0.1):
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


    def smooth_loss_fn(self, model, z, noise_std=0.1):
        """
        Smoothness loss: z에 작은 노이즈를 더했을 때 예측이 크게 변하지 않도록.
        """
        z_noisy = z + noise_std * torch.randn_like(z)
        
        cost_original = model.predict_cost(z)
        cost_noisy = model.predict_cost(z_noisy)
        
        smooth_loss = F.mse_loss(cost_original, cost_noisy)
        return smooth_loss


    def kld_loss_fn(self, mean, logvar):
        """
        KL Divergence: q(z|x) || N(0, I)
        """
        kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return kld


    def compute_total_loss(self, model, segment_sizes, features, labels, config, return_components=False):
        """
        Total loss 계산 (Segment 기반 데이터용).
        total_loss = reg_loss + λ_pair * pair_loss + γ * smooth_loss + β * kld_loss
        """
        # Forward pass
        cost_pred, mean, logvar, z = model(segment_sizes, features, use_mean=True)
        
        # Individual losses
        reg = self.reg_loss_fn(cost_pred, labels, loss_type=config.get('loss_type', 'mse'))
        pair = self.pair_loss_fn(cost_pred.view(-1), labels.view(-1), margin=config.get('margin', 0.1))
        smooth = self.smooth_loss_fn(model, z, noise_std=config.get('noise_std', 0.1))
        kld = self.kld_loss_fn(mean, logvar)
        
        # Weighted sum
        total = reg + config.get('lambda_pair', 0.1) * pair + config.get('gamma', 0.01) * smooth + config.get('beta', 0.001) * kld
        
        if return_components:
            return total, {
                'reg_loss': reg.item(),
                'pair_loss': pair.item(),
                'smooth_loss': smooth.item(),
                'kld_loss': kld.item(),
                'total_loss': total.item(),
            }
        
        return total