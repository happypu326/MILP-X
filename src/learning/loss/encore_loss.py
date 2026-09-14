import torch
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer


class EnCoreLossComputer(BaseLossComputer):
    """Plain per-variable BCE against the early-to-final consistency label
    y_i = 1[x*_i == x_ES_i], over binary variables only (EnCore).

    model_output : per-graph-variable consistency probability (already sigmoid'd),
                   concatenated across the batch.
    Uses batch.consistency_labels (per-variable, concatenated) and
    batch.varInds / batch.ntvars to slice per graph and select binary vars.
    """

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        cons_all = batch.consistency_labels
        n_graphs = batch.nsols.shape[0]
        loss_list = []
        pred_arrow = 0
        lab_arrow = 0
        for i in range(n_graphs):
            n_var = int(batch.ntvars[i])
            b_vars = batch.varInds[i][1][0].long()
            pred = model_output[pred_arrow:pred_arrow + n_var].squeeze()[b_vars]
            labels = cons_all[lab_arrow:lab_arrow + n_var][b_vars]
            pred_arrow += n_var
            lab_arrow += n_var
            bce = -(labels * (pred + 1e-8).log()
                    + (1 - labels) * (1 - pred + 1e-8).log())
            loss_list.append(bce.mean())
        return torch.stack(loss_list), {}
