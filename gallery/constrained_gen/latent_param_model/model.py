from __future__ import annotations

from dataclasses import dataclass

import copy
import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
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


class PerLayerCrossAttentionDecoder(nn.Module):
    def __init__(self, dec_layer: nn.TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dec_layer.self_attn.embed_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            # each layer does: self-attn -> cross-attn(memory) -> FFN
            x = layer(
                tgt=x,
                memory=memory,
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
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)

        # explicit per-layer cross-attention decoder
        self.decoder = PerLayerCrossAttentionDecoder(dec_layer, num_layers=cfg.num_decoder_layers)

        # replace masked mean with attention-weighted mean pooling
        self.pool = AttentionMeanPool(d_model)

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
        # self.cost_head = nn.Linear(cfg.latent_dim, 1)
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
        decoder_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self._embed(decoder_input_ids, decoder_var_ids)
        seq_len = y.size(1)
        causal_mask = self._causal_mask(seq_len, y.device)
        dec = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_pad_mask,
        )
        return self.lm_head(dec)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        logits = self.decode(decoder_input_ids, decoder_var_ids, memory, dec_pad)
        cost_pred = self.cost_head(z).squeeze(-1)
        return ModelOutput(
            logits=logits,
            mu=mu,
            logvar=logvar,
            z=z,
            cost_pred=cost_pred,
        )
