"""H1 Graph-Q-Former.

Replaces the baseline's `MLP -> mean(dim=0)` graph-to-LLM bridge with a
small cross-attention block that pools a variable number of retrieved
subgraph embeddings into a fixed number of soft prompt tokens fed to the
LLM. Inspired by BLIP-2 / Q-Former and adapted for the per-sample graph
list produced by `retrieve.py:retrieval_topk`.
"""
import torch
import torch.nn as nn


class GraphQFormer(nn.Module):
    """Cross-attention pooling from N subgraph embeddings to q query tokens.

    Inputs at forward time:
        subgraph_embeds: tensor (N, gnn_dim) — the GNN-encoded, scatter-mean
            pooled embedding of each retrieved subgraph for one sample.

    Output:
        tokens: tensor (q, llm_dim) — soft-prompt tokens prepended to the
            LLM input. q is fixed regardless of N.
    """

    def __init__(
        self,
        gnn_dim: int,
        llm_dim: int,
        num_query_tokens: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.llm_dim = llm_dim

        # learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, llm_dim) * 0.02
        )

        # project subgraph embeds (gnn_dim) into llm_dim so they can be the
        # K/V of cross-attention. Mirrors the original projector shape.
        self.in_proj = nn.Sequential(
            nn.Linear(gnn_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, llm_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim),
        )

    def forward(self, subgraph_embeds: torch.Tensor) -> torch.Tensor:
        # (N, gnn_dim) -> (1, N, llm_dim)
        kv = self.in_proj(subgraph_embeds).unsqueeze(0)
        # (q, llm_dim) -> (1, q, llm_dim)
        q = self.query_tokens.unsqueeze(0)

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0)  # (q, llm_dim)
