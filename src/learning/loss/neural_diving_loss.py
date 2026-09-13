"""
SelectiveNet selective loss for the Neural Diving predictor.

Per instance, over the binary variables B:
  * target marginal m_j = sum_q w_q x^{(q)}_j with Boltzmann weights
    w_q propto exp(-obj_q / weight_norm) (same energy weighting as PS);
  * per-variable BCE  l_j = BCE(p_j, m_j);
  * selective risk    R = sum_j g_j l_j / sum_j g_j;
  * coverage penalty  lambda * max(0, target_coverage - mean(g))^2;
  * auxiliary term     alpha * mean(l_j)  (SelectiveNet auxiliary head that keeps
    the predictor calibrated on *all* variables, not just the selected ones).

L = R + lambda * penalty + alpha * mean(l_j).
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM


class NeuralDivingLossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_coverage = config.get('target_coverage', 0.7)
        self.lam = config.get('coverage_lambda', 16.0)
        self.alpha = config.get('aux_alpha', 0.5)
        self.eps = 1e-8

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        pred_logit, sel_logit = model_output
        group = batch.group
        solInd = batch.nsols

        target_sols, target_vals = [], []
        solEndInd = valEndInd = 0
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])
            solStartInd, solEndInd = solEndInd, solInd[i] * nvar + solEndInd
            valStartInd, valEndInd = valEndInd, valEndInd + solInd[i]
            target_sols.append(batch.solutions[solStartInd:solEndInd].reshape(-1, nvar))
            target_vals.append(batch.objVals[valStartInd:valEndInd])

        loss_list = []
        index_arrow = 0
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            group_ind = group[ind].cpu().item()
            weight_norm = ENERGY_WEIGHT_NORM.get(GROUP_CLASS[group_ind], 1)
            varInds = batch.varInds[ind]
            varname_map = varInds[0][0]
            b_vars = varInds[1][0].long()
            sols_b = sols[:, varname_map][:, b_vars]                    # [S, |B|]

            # energy-weighted target marginal per binary
            w = torch.softmax(-vals / weight_norm, dim=0)               # [S]
            m = (w[:, None] * sols_b).sum(dim=0).clamp(0, 1)            # [|B|]

            n_var = int(batch.ntvars[ind])
            p_logit = pred_logit[index_arrow:index_arrow + n_var].squeeze()[b_vars]
            g = sel_logit[index_arrow:index_arrow + n_var].squeeze()[b_vars].sigmoid()
            index_arrow += n_var

            if b_vars.numel() == 0:
                continue

            ell = F.binary_cross_entropy_with_logits(p_logit, m, reduction='none')  # [|B|]
            selective_risk = (g * ell).sum() / (g.sum() + self.eps)
            coverage = g.mean()
            penalty = torch.clamp(self.target_coverage - coverage, min=0) ** 2
            aux = ell.mean()

            loss_list.append(selective_risk + self.lam * penalty + self.alpha * aux)

        if not loss_list:
            zero = pred_logit.sum() * 0.0
            return zero.reshape(1), {}
        batch_loss = torch.stack(loss_list)
        return batch_loss, {'nd_loss': batch_loss.detach().mean().item()}
