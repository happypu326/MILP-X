"""
Neural Diving evaluator: predict marginals + selection gate, then dive
(coverage-based hard fixing by the learned gate) and solve the residual sub-MIP.

Reuses PSFamilyEvaluator's trust-region solve; the gate is written into a 6th
element of each score row so the base class's `_fix_dive` prioritises variables
by learned selection confidence instead of raw prediction confidence.
"""

import torch
from typing import List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features


class NeuralDivingEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config):
        # force diving fix strategy
        config = dict(config)
        config.setdefault("fix_strategy", "dive")
        super().__init__(model, config)

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)

        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices, A._values(), use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0],
        )
        batch_indices = torch.zeros(v_nodes.shape[0], dtype=torch.long)

        pred_logit, sel_logit = self.model(
            constraint_features.to(self.device),
            edge_indices.to(self.device),
            edge_features.to(self.device),
            v_nodes.to(self.device),
            batch_indices.to(self.device),
        )
        prob = pred_logit.sigmoid().cpu().squeeze()
        gate = sel_logit.sigmoid().cpu().squeeze()

        all_varname = list(v_map.keys())
        binary_name = {all_varname[i] for i in b_vars.tolist()}
        scores = []
        for i in range(len(v_map)):
            vtype = 'BINARY' if all_varname[i] in binary_name else 'C'
            # 6th element = learned selection gate (used by _fix_dive)
            scores.append([i, all_varname[i], prob[i].item(), -1, vtype, gate[i].item()])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores
