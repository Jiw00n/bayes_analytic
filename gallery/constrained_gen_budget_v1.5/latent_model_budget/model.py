from __future__ import annotations

from dataclasses import dataclass

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    memory: torch.Tensor
    cost_pred: torch.Tensor | None


class AttentionMeanPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], pad_mask: [B, L] where True means PAD
        attn_logits = self.score(x).squeeze(-1)  # [B, L]
        attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, D]


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaptiveDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, latent_dim: int, adaln: bool=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.act = nn.GELU()
        self.adaln = adaln
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 9 * d_model),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        latent_cond: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.adaln:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mca,
                scale_mca,
                gate_mca,
                shift_ff,
                scale_ff,
                gate_ff,
            ) = self.ada_mod(latent_cond).chunk(9, dim=-1)

            x = tgt

            x_norm = _modulate(self.norm1(x), shift_msa, scale_msa)
            self_attn_out, _ = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )
            x = x + gate_msa.unsqueeze(1) * self.dropout(self_attn_out)

            x_norm = _modulate(self.norm2(x), shift_mca, scale_mca)
            cross_attn_out, _ = self.cross_attn(
                x_norm,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )
            x = x + gate_mca.unsqueeze(1) * self.dropout(cross_attn_out)

            x_norm = _modulate(self.norm3(x), shift_ff, scale_ff)
            ff_out = self.linear2(self.dropout_ff(self.act(self.linear1(x_norm))))
            x = x + gate_ff.unsqueeze(1) * self.dropout(ff_out)
            return x

        elif not self.adaln:

            del latent_cond

            x = tgt

            x_norm = self.norm1(x)
            self_attn_out, _ = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(self_attn_out)

            x_norm = self.norm2(x)
            cross_attn_out, _ = self.cross_attn(
                x_norm,
                memory,
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(cross_attn_out)

            x_norm = self.norm3(x)
            ff_out = self.linear2(self.dropout_ff(self.act(self.linear1(x_norm))))
            x = x + self.dropout(ff_out)
            return x


class PerLayerCrossAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int, latent_dim: int, adaln: bool):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AdaptiveDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    latent_dim=latent_dim,
                    adaln=adaln,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        latent_cond: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                tgt=x,
                memory=memory,
                latent_cond=latent_cond,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


class LatentParamVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_vars: int,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.num_vars = num_vars

        d_model = cfg.d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.var_emb = nn.Embedding(num_vars, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)

        self.decoder = PerLayerCrossAttentionDecoder(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_layers=cfg.num_decoder_layers,
            latent_dim=cfg.latent_dim,
            adaln=cfg.adaln,
        )

        # replace masked mean with attention-weighted mean pooling
        self.pool = AttentionMeanPool(d_model)
        
        # mean pool
        # self.pool = lambda x, pad_mask: (
        #     (x * (~pad_mask).unsqueeze(-1).to(x.dtype)).sum(dim=1)
        #     / (~pad_mask).sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        # )


        # # cls token pooling
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.to_mu = nn.Linear(d_model, cfg.latent_dim)
        self.to_logvar = nn.Linear(d_model, cfg.latent_dim)
        self.latent_to_memory = nn.Sequential(
            nn.Linear(cfg.latent_dim, d_model * cfg.latent_token_count),
            # nn.GELU(),
            # nn.Linear(d_model * cfg.latent_token_count, d_model * cfg.latent_token_count),
        )

        self.cost_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.dim_feedforward),
            nn.GELU(),
            nn.Linear(cfg.dim_feedforward, 1),
        )
        # self.cost_w = nn.Parameter(torch.randn(cfg.latent_dim) * 0.02)
        # self.cost_bias = nn.Parameter(torch.zeros(()))
        # self.cost_gamma = nn.Parameter(torch.zeros(()))
        # self.cost_head = self.predict_cost

        self.lm_head = nn.Linear(d_model, vocab_size)

    def _positions(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    def _embed(self, token_ids: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        pos = self._positions(bsz, seq_len, token_ids.device)
        x = self.token_emb(token_ids) + self.var_emb(var_ids) + self.pos_emb(pos)
        # x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.dropout(x)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def encode(
        self,
        encoder_token_ids: torch.Tensor,
        encoder_var_ids: torch.Tensor,
        encoder_pad_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._embed(encoder_token_ids, encoder_var_ids)
        enc = self.encoder(x, src_key_padding_mask=encoder_pad_mask)
        pooled = self.pool(enc, encoder_pad_mask)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        memory = self.latent_to_memory(z).view(z.size(0), self.cfg.latent_token_count, self.cfg.d_model)
        return mu, logvar, z, memory


    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_var_ids: torch.Tensor,
        memory: torch.Tensor,
        latent_cond: torch.Tensor,
        decoder_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self._embed(decoder_input_ids, decoder_var_ids)
        seq_len = y.size(1)
        causal_mask = self._causal_mask(seq_len, y.device)
        dec = self.decoder(
            tgt=y,
            memory=memory,
            latent_cond=latent_cond,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_pad_mask,
        )
        return self.lm_head(dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # def predict_cost(self, z: torch.Tensor) -> torch.Tensor:
    #     gamma = F.softplus(self.cost_gamma) + 1e-6
    #     w = self.cost_w / (self.cost_w.norm() + 1e-12)
    #     score = gamma * (z @ w) + self.cost_bias
    #     return score

    def forward(
        self,
        encoder_token_ids: torch.Tensor,
        encoder_var_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_var_ids: torch.Tensor,
        pad_token_id: int,
    ) -> ModelOutput:
        enc_pad = encoder_token_ids.eq(pad_token_id)
        dec_pad = decoder_input_ids.eq(pad_token_id)
        mu, logvar, z, memory = self.encode(encoder_token_ids, encoder_var_ids, enc_pad)
        logits = self.decode(decoder_input_ids, decoder_var_ids, memory, z, dec_pad)
        cost_pred = self.cost_head(mu)
        return ModelOutput(
            logits=logits,
            mu=mu,
            logvar=logvar,
            z=z,
            memory=memory,
            cost_pred=cost_pred,
        )


    # cls token prepend + transformer pooling
    # def encode(
    #     self,
    #     encoder_token_ids: torch.Tensor,
    #     encoder_var_ids: torch.Tensor,
    #     encoder_pad_mask: torch.Tensor,
    #     deterministic: bool = False,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     bsz, seq_len = encoder_token_ids.shape
    #     device = encoder_token_ids.device

    #     # 원래 토큰들 임베딩
    #     body_pos = self._positions(bsz, seq_len + 1, device)[:, 1:]
    #     body = (
    #         self.token_emb(encoder_token_ids)
    #         + self.var_emb(encoder_var_ids)
    #         + self.pos_emb(body_pos)
    #     )

    #     # CLS 토큰 prepend
    #     cls_pos = self._positions(bsz, seq_len + 1, device)[:, :1]
    #     cls = self.cls_token.expand(bsz, 1, -1) + self.pos_emb(cls_pos)

    #     x = torch.cat([cls, body], dim=1)
    #     x = self.dropout(x)

    #     cls_pad = torch.zeros((bsz, 1), dtype=torch.bool, device=device)
    #     enc_pad = torch.cat([cls_pad, encoder_pad_mask], dim=1)

    #     enc = self.encoder(x, src_key_padding_mask=enc_pad)

    #     pooled = enc[:, 0, :]   # CLS pooling
    #     mu = self.to_mu(pooled)
    #     logvar = self.to_logvar(pooled)
    #     z = mu if deterministic else self.reparameterize(mu, logvar)
    #     memory = self.latent_to_memory(z).view(z.size(0), self.cfg.latent_token_count, self.cfg.d_model)
    #     return mu, logvar, z, memory

