"""
Dual-head bipartite GNN for Constraint Matters: predicts BOTH a per-variable
assignment marginal and a per-constraint criticality (critical-tight-constraint)
probability from the shared constraint-variable encoder.

This is the dependency-light core of the paper (no T5 abstract-graph modality);
the encoder is the standard Gasse half-convolution, which already returns both
variable and constraint node embeddings, so only a second read-out head is
needed.
"""

import torch

from .gcn import GNNEncoder, GNNDecoder


class CTCPolicy(torch.nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2, variable_nfeats=6):
        super().__init__()
        self.encoder = GNNEncoder(emb_size, constraint_nfeats, edge_nfeats, variable_nfeats)
        self.var_head = GNNDecoder(emb_size)                 # per-variable logit
        self.con_head = torch.nn.Sequential(                 # per-constraint logit
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        out = self.encoder(constraint_features, edge_indices, edge_features,
                            variable_features, batch_indices=batch_indices)
        var_emb, con_emb = out[0], out[1]
        var_logit = self.var_head(var_emb)                   # [n_var]
        con_logit = self.con_head(con_emb).squeeze(-1)       # [n_con]  (incl. obj node)
        return var_logit, con_logit
