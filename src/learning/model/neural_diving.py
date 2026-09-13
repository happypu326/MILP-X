"""
Neural Diving predictor (Nair et al., "Solving Mixed Integer Programs Using
Neural Networks", 2020) with a SelectiveNet coverage head.

The bipartite GNN produces, per variable:
  * a prediction logit -> Bernoulli marginal p_j (energy model over solutions),
  * a selection gate g_j in [0,1] (SelectiveNet, Geifman & El-Yaniv 2019) that
    decides which variables the model is confident enough to assign.

At test time the high-gate variables are fixed ("dived") and the residual
sub-MIP is solved -- i.e. coverage-controlled hard fixing.
"""

import torch
import torch.nn as nn

from .gcn import GNNEncoder


class NeuralDivingPolicy(nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6):
        super().__init__()
        self.encoder = GNNEncoder(emb_size, constraint_nfeats, edge_nfeats, variable_nfeats)
        self.pred_head = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )
        self.sel_head = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        enc = self.encoder(constraint_features, edge_indices, edge_features,
                           variable_features, batch_indices=batch_indices)
        var_emb = enc[0] if isinstance(enc, tuple) else enc
        pred_logit = self.pred_head(var_emb).squeeze(-1)
        sel_logit = self.sel_head(var_emb).squeeze(-1)
        return pred_logit, sel_logit
