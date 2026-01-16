import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class VAE_feature_head(nn.Module):
    def __init__(self, input_dim, feature_dim=None, latent_dim=16, hidden_dim=128):
        """
        input_dim: 2 * D (v_norm + is_zero concat한 차원)
        latent_dim: latent space 차원
        hidden_dim: MLP hidden 크기
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            
            # 출력은 연속값이니까 activation 없이 그대로
        )

        if feature_dim is None:
            self.use_feature = False
        else:
            self.use_feature = True
            self.feature_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),  # features.shape[1]는 feature 차원
            )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict_feature(self, z):
        return self.feature_predictor(z)

    def forward(self, x, use_mean=True):
        mu, logvar = self.encode(x)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        if self.use_feature:
            feature_pred = self.predict_feature(z)
        else:
            feature_pred = None
        return x_recon, mu, logvar, z, feature_pred




class VAECostPredictor(nn.Module):
    """
    VAE 기반 Cost Regression 모델
    
    구조:
    - input → segment_encoder → segment_sum → VAE encoder → z → cost_predictor → cost
    
    특징:
    - Pretrained VAE encoder를 finetune (작은 learning rate)
    - Cost predictor는 더 큰 learning rate로 학습
    - 전체 forward 경로가 완전히 미분 가능 (detach, stop_grad 없음)
    """
    
    def __init__(self, input_dim, feature_dim=None, hidden_dim=256, latent_dim=64, 
                 predictor_hidden=256, predictor_layers=2, dropout=0.1, use_feature=False):
        super(VAECostPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========== Cost Predictor (새로 학습) ==========
        predictor_modules = []
        current_dim = latent_dim
        for i in range(predictor_layers):
            predictor_modules.extend([
                nn.Linear(current_dim, predictor_hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity(),
            ])
            current_dim = predictor_hidden
        predictor_modules.append(nn.Linear(predictor_hidden, 1))
        
        self.cost_predictor = nn.Sequential(*predictor_modules)

        self.use_feature = use_feature
        if self.use_feature:
            pass
            self.feature_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),  # feature_dim는 feature 차원
            )
        
    
    def encode(self, input_data, use_mean=False):
        """
        Full encoding path: features → z
        완전히 미분 가능
        """
                
        # VAE Encoder
        h = self.encoder(input_data)
        
        mean = self.fc_mu(h)
        if not use_mean:
            logvar = self.fc_logvar(h)
        else:
            return mean
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick - 미분 가능"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def predict_cost(self, z):
        """z → cost prediction - 완전히 미분 가능"""
        return self.cost_predictor(z).squeeze(-1)
    
    def predict_feature(self, z):
        return self.feature_predictor(z)
    
    def forward(self, input_data, use_mean=True):
        """
        Forward pass: input → z → cost
        
        Args:
            use_mean: True면 reparameterize 대신 mean 사용 (inference용)
        
        Returns:
            cost_pred: 예측된 cost
            mean: latent mean
            logvar: latent log-variance
            z: sampled/mean latent vector
        """
        mean, logvar = self.encode(input_data)
        
        if use_mean:
            z = mean  # Inference시 deterministic
        else:
            z = self.reparameterize(mean, logvar)  # Training시 stochastic
        
        cost_pred = self.predict_cost(z)
        
        return cost_pred, mean, logvar, z
    
    def get_encoder_params(self):
        """Encoder 파라미터 (작은 lr)"""
        encoder_params = []
        encoder_params.extend(self.encoder.parameters())
        encoder_params.extend(self.fc_mu.parameters())
        encoder_params.extend(self.fc_logvar.parameters())
        return encoder_params
    
    def get_cost_predictor_params(self):
        """Predictor 파라미터 (큰 lr)"""
        return self.cost_predictor.parameters()
    
    def get_feature_predictor_params(self):
        """Feature Predictor 파라미터"""
        return self.feature_predictor.parameters()

    def load_pretrained_encoder(self, checkpoint):
        """Pretrained VAE encoder 가중치 로드"""
        

        vae_state = checkpoint
        
        # 매칭되는 키만 로드
        encoder_keys = ['encoder', 'fc_mu', 'fc_logvar']
        own_state = self.state_dict()
        
        loaded_keys = []
        for name, param in vae_state.items():
            if any(name.startswith(k) for k in encoder_keys):
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded_keys.append(name)
        
        # print(f"Loaded {len(loaded_keys)} parameters from pretrained VAE")
        # return loaded_keys

    def _enable_dropout(self):
        """모든 Dropout 모듈을 train 모드로 강제 활성화"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def mc_predict(self, input_tensor, T=20):
        """
        MC Dropout 기반 불확실성 추정
        
        Args:
            input_tensor: 입력 텐서 (shape [N, input_dim])
            T: MC 샘플 수
        
        Returns:
            mean: epistemic 평균 cost (shape [N])
            var: epistemic 분산 (shape [N])
        """

        self.eval()  # 전체 모델을 eval 모드로
        self._enable_dropout()  # Dropout만 train 모드로 활성화
        
        
        with torch.no_grad():
            predictions = []
            
            for _ in range(T):
                # Encode
                z, logvar = self.encode(input_tensor)
                cost_pred = self.predict_cost(z)
                predictions.append(cost_pred)
            
            predictions = torch.stack(predictions, dim=0)
            
            # epistemic mean & variance
            mc_mean = predictions.mean(dim=0)
            mc_var = predictions.var(dim=0)

        return mc_mean, mc_var


def make_vae_reg_model(vae, hyperparameter, input_dim, latent_dim, hidden_dim, y_std, device, verbose=True):


    cnt = 0
    for vals in itertools.product(*hyperparameter.values()):
        (lambda_reg, lambda_pair, margin_scale, gamma, beta, noise_std, 
        encoder_lr, feature_predictor_lr, cost_predictor_lr,  epochs) = vals
        cnt += 1
        if verbose:
            print(f"Experiment {cnt}/{len(list(itertools.product(*hyperparameter.values())))}")
            print(f"lambda_reg={lambda_reg}, lambda_pair={lambda_pair}, margin_scale={margin_scale}, \
              gamma={gamma}, beta={beta}, noise_std={noise_std}\nencoder_lr={encoder_lr}, cost_predictor_lr={cost_predictor_lr}, epochs={epochs}")
        config = {
                    'encoder_lr': encoder_lr,
                    'feature_predictor_lr': feature_predictor_lr,
                    'cost_predictor_lr': cost_predictor_lr,
                    'lambda_reg' : lambda_reg,
                    'lambda_pair': lambda_pair,
                    'gamma': gamma,
                    'beta': beta,
                    'margin': margin_scale * y_std,
                    'noise_std': noise_std,
                    'loss_type': 'mse',
                    'epochs': epochs,
                }

        vae_cost_model = VAECostPredictor(input_dim=input_dim, 
                                    latent_dim=latent_dim, 
                                    hidden_dim=hidden_dim, 
                                    predictor_layers=2,
                                    dropout=0.1, use_feature=False).to(device)
        vae_cost_model.load_pretrained_encoder(vae.state_dict())
        optimizer = torch.optim.AdamW([
            {'params': vae_cost_model.get_encoder_params(), 'lr': config['encoder_lr']},
            {'params': vae_cost_model.get_cost_predictor_params(), 'lr': config['cost_predictor_lr']}
        ], weight_decay=1e-5)
    return vae_cost_model, optimizer, config