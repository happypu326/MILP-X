"""
SHSP: Structure-Aware Hierarchical Solution Prediction (arXiv 2608.25282).

Replaces one-shot marginal decoding with hierarchical *conditional* decoding:
variables are decoded along a hierarchy of increasing coupling strength, each
level conditioned on previously assigned variables. The GNN is made conditional
by two extra variable channels [known_mask, known_value]; it is trained with
random reveal (masked prediction) so it learns p(x_j | already-assigned).

This module defines the conditional predictor. The hierarchical decoding order
and confidence-aware mask-and-repair live in the SHSP evaluator.
"""

import torch
import torch.nn as nn

from .bipartite_encoders import GasseGNNPolicy


class SHSPPolicy(nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, jumping_knowledge=True):
        super().__init__()
        # two extra variable channels: known_mask in {0,1}, known_value in {0,1}
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + 2,
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, known_mask, known_value,
                batch_indices=None, is_training=False):
        v_aug = torch.cat(
            [variable_features, known_mask.reshape(-1, 1), known_value.reshape(-1, 1)],
            dim=-1,
        )
        return self.backbone(constraint_features, edge_indices, edge_features,
                             v_aug, batch_indices, is_training)
