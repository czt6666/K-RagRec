"""H2 Retrieval Gate.

Replaces the static popularity-threshold rule with a small MLP that scores
each retrieved subgraph for relevance to the user query, allowing the model
to learn end-to-end which retrievals to trust. Gradient flows through a
soft sigmoid weight at aggregation time.
"""
import torch
import torch.nn as nn


class RetrievalGate(nn.Module):
    """Score N retrieved subgraphs against a single query embedding.

    Args:
        emb_dim: dim of subgraph and query embeddings (1024 default for SBERT).
        hidden: width of the gate MLP.

    Forward inputs:
        subgraph_embeds: (N, emb_dim) GNN-pooled embedding per subgraph.
        query_emb:        (emb_dim,) SBERT embedding of the user history /
                          recommendation prompt.

    Returns:
        weights: (N,) sigmoid-activated scalar weight per subgraph in [0, 1].
        logits:  (N,) raw scores (useful for entropy/sparsity regularizers).
    """

    def __init__(self, emb_dim: int = 1024, hidden: int = 256):
        super().__init__()
        # Concatenated [subgraph, query, |subgraph - query|, subgraph * query]
        # gives the gate a richer comparison feature than plain concatenation.
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, subgraph_embeds: torch.Tensor, query_emb: torch.Tensor):
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        q = query_emb.expand(subgraph_embeds.size(0), -1)
        feat = torch.cat([
            subgraph_embeds,
            q,
            (subgraph_embeds - q).abs(),
            subgraph_embeds * q,
        ], dim=-1)
        logits = self.mlp(feat).squeeze(-1)
        weights = torch.sigmoid(logits)
        return weights, logits

    @staticmethod
    def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = False):
        """Differentiable 0/1 mask via Gumbel-Sigmoid (straight-through if hard=True)."""
        if not torch.is_grad_enabled():
            return (torch.sigmoid(logits) > 0.5).float()
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        gumbel = torch.log(u) - torch.log1p(-u)
        soft = torch.sigmoid((logits + gumbel) / tau)
        if not hard:
            return soft
        hard_mask = (soft > 0.5).float()
        return hard_mask + soft - soft.detach()  # straight-through estimator
