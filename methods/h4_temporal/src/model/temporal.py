"""H4 Sequential / temporal encoder.

The baseline ignores the order of the user's 10 history items: every
retrieved subgraph is mean-pooled into one soft token, scrubbing the
sequential signal that SASRec etc. rely on. This module re-injects the
sequence by encoding the raw item-id history through a small Transformer
with learned positional embeddings, producing one extra soft prompt token
that the LLM sees alongside the graph soft prompt.

If a per-interaction timestamp tensor is available (looked up from
``ratings_45.txt`` upstream), it is bucketed and added as a second
positional signal so very recent vs very old interactions get distinct
embeddings.
"""
import math

import torch
import torch.nn as nn


class SequentialEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4000,
        max_seq_len: int = 32,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 1024,
        num_time_buckets: int = 16,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = nn.Embedding(num_time_buckets + 1, d_model)  # +1 for "unknown"
        self.num_time_buckets = num_time_buckets
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _bucket_dt(self, dt_seconds: torch.Tensor) -> torch.Tensor:
        """Map a real-time delta (seconds) into a discrete bucket id.

        Buckets follow log-spaced cutoffs (1h, 6h, 1d, 1w, 1mo, 6mo, 1y, ...)
        which works for movie-rating timescales. Unknown/negative dt -> 0
        (a reserved "unknown" id).
        """
        if dt_seconds is None:
            return None
        cutoffs = torch.tensor([
            0, 3600, 21600, 86400, 604800, 2592000, 15552000, 31536000,
            63072000, 126144000, 252288000, 504576000, 1009152000,
            2018304000, 4036608000, 8073216000.0,
        ], device=dt_seconds.device)
        # bucket id = number of cutoffs strictly less than dt; clamp to range
        b = (dt_seconds.unsqueeze(-1) >= cutoffs).long().sum(dim=-1)
        b = b.clamp(min=1, max=self.num_time_buckets)  # reserve 0 for unknown
        return b

    def forward(
        self,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        item_ids:    (B, L) long, padding_idx=0
        time_deltas: (B, L) float seconds since "now"; None -> ignore
        mask:        (B, L) bool, True for valid positions; None -> infer from item_ids != 0

        Returns: (B, out_dim) — one summary token per sample.
        """
        if mask is None:
            mask = item_ids != 0
        B, L = item_ids.shape

        x = self.item_emb(item_ids)  # (B, L, d)
        positions = torch.arange(L, device=item_ids.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        if time_deltas is not None:
            tb = self._bucket_dt(time_deltas)
            x = x + self.time_emb(tb)

        # prepend a CLS token so we can pool a single summary
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        # transformer expects key_padding_mask where True = MASKED OUT
        key_padding = ~full_mask
        x = self.encoder(x, src_key_padding_mask=key_padding)
        out = self.out_proj(x[:, 0])  # CLS pooled
        return out
