"""
Dual-head focal loss for Constraint Matters:
    L = lambda * focal(variable assignment) + (1 - lambda) * focal(constraint criticality)
Both heads use a binary focal loss (Lin et al. 2017): alpha-weighted,
(1 - p_t)^gamma-modulated BCE. Variable loss is over binary variables; constraint
loss is over all constraint nodes (the appended objective node has label 0).
"""

import torch
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer


class DualFocalLossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.lam = config.get('ctc_lambda', 0.4)
        self.alpha = config.get('focal_alpha', 0.75)
        self.gamma = config.get('focal_gamma', 2.0)

    def _focal(self, p, y):
        p = p.clamp(1e-6, 1 - 1e-6)
        pt = torch.where(y > 0.5, p, 1 - p)
        alpha_t = torch.where(y > 0.5, self.alpha, 1 - self.alpha)
        return -(alpha_t * (1 - pt).pow(self.gamma) * pt.log())

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        var_prob, con_prob = model_output          # both already sigmoid'd
        n_graphs = batch.nsols.shape[0]
        losses = []
        v_arrow, c_arrow = 0, 0
        for i in range(n_graphs):
            n_var = int(batch.ntvars[i])
            n_con = int(batch.ntcons[i])
            b_vars = batch.varInds[i][1][0].long()

            vp = var_prob[v_arrow:v_arrow + n_var].squeeze()[b_vars]
            vy = batch.var_labels[v_arrow:v_arrow + n_var][b_vars]
            cp = con_prob[c_arrow:c_arrow + n_con].squeeze()
            cy = batch.ctc_labels[c_arrow:c_arrow + n_con]
            v_arrow += n_var
            c_arrow += n_con

            l_v = self._focal(vp, vy).mean()
            l_c = self._focal(cp, cy).mean()
            losses.append(self.lam * l_v + (1 - self.lam) * l_c)
        return torch.stack(losses), {}
