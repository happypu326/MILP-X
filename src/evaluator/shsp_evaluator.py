"""
SHSP evaluator: structure-aware hierarchical conditional decoding.

1. Build a variable coupling signal from the constraint structure (how many
   other binaries each binary competes with across its constraints).
2. Order binaries into K hierarchy levels of increasing coupling strength and
   decode level-by-level, conditioning each level on previously assigned
   variables (fed back through the known_mask / known_value channels).
3. Confidence-aware mask-and-repair: free the least-confident assignments and
   re-predict once to correct error accumulation.
4. Emit the final marginals as scores for the trust-region search.
"""

import torch
from typing import List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features


class SHSPEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.n_levels = config.get("n_levels", 4)
        self.repair_frac = config.get("repair_frac", 0.2)   # free this frac (least confident) then repair

    def _coupling_order(self, edge_indices, b_vars, n_total):
        """Return binary-variable column indices ordered by ascending coupling
        strength (least-coupled decoded first)."""
        device = edge_indices.device
        bin_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        bin_mask[b_vars] = True
        c_ids, v_ids = edge_indices[0], edge_indices[1]
        keep = bin_mask[v_ids]
        c_ids, v_ids = c_ids[keep], v_ids[keep]
        ncons = int(edge_indices[0].max()) + 1 if edge_indices.numel() else 1
        bindeg = torch.zeros(ncons, device=device).scatter_add(
            0, c_ids, torch.ones_like(c_ids, dtype=torch.float))
        coupling = torch.zeros(n_total, device=device).scatter_add(
            0, v_ids, (bindeg[c_ids] - 1).clamp(min=0))
        cval = coupling[b_vars]
        order = torch.argsort(cval)          # ascending
        return b_vars[order]

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        device = self.device
        cons_f = c_nodes.cpu(); cons_f[torch.isnan(cons_f)] = 1
        cons_f = cons_f.to(device)
        edge_idx = A._indices().to(device)
        edge_feat = build_edge_features(A._indices(), A._values(),
                                        use_edge_coeff=self.use_edge_coeff,
                                        num_cons=c_nodes.shape[0]).to(device)
        var_f = v_nodes.to(device)
        n_total = v_nodes.shape[0]
        b_vars = b_vars.to(device).long()
        batch = torch.zeros(n_total, dtype=torch.long, device=device)

        known_mask = torch.zeros(n_total, device=device)
        known_value = torch.zeros(n_total, device=device)

        def predict():
            logits = self.model(cons_f, edge_idx, edge_feat, var_f,
                                 known_mask, known_value, batch)
            return logits.sigmoid()

        ordered = self._coupling_order(edge_idx, b_vars, n_total)
        levels = torch.chunk(ordered, max(1, self.n_levels))

        # hierarchical conditional decoding
        for lvl in levels:
            p = predict()
            known_value[lvl] = (p[lvl] >= 0.5).float()
            known_mask[lvl] = 1.0

        # confidence-aware mask-and-repair: free least-confident assignments,
        # then re-predict once to correct them.
        p = predict()
        conf = (p[b_vars] - 0.5).abs()
        n_free = int(round(self.repair_frac * b_vars.numel()))
        if n_free > 0:
            worst = b_vars[torch.argsort(conf)[:n_free]]
            known_mask[worst] = 0.0
        p_final = predict().cpu()

        BD = torch.full((n_total,), 0.5)
        BD[b_vars.cpu()] = p_final[b_vars.cpu()]
        all_varname = list(v_map.keys())
        binary_name = {all_varname[i] for i in b_vars.tolist()}
        scores = []
        for i in range(len(v_map)):
            vtype = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], BD[i].item(), -1, vtype])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores
