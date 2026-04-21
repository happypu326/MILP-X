import torch
import torch.nn as nn
import torch.nn.functional as F
from src.learning.loss.base_loss import BaseLossComputer
from typing import Tuple
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM

class BCELossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        group = batch.group
        
        solInd = batch.nsols
        target_sols = []
        target_vals = []
        solEndInd = 0
        valEndInd = 0
        
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])
            solStartInd = solEndInd
            solEndInd = solInd[i] * nvar + solStartInd
            valStartInd = valEndInd
            valEndInd = valEndInd + solInd[i]
            sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
            vals = batch.objVals[valStartInd:valEndInd]
            
            target_sols.append(sols)
            target_vals.append(vals)
        
        loss_list = []
        index_arrow = 0
        
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            group_ind = group[ind].cpu().item()
            problem_ind = GROUP_CLASS[group_ind]
            weight_norm = ENERGY_WEIGHT_NORM.get(problem_ind, 1)
            
            n_vals = vals
            exp_weight = torch.exp(-n_vals / weight_norm)
            weight = exp_weight / exp_weight.sum()
            
            varInds = batch.varInds[ind]
            varname_map = varInds[0][0]
            b_vars = varInds[1][0].long()
            
            sols = sols[:, varname_map][:, b_vars]
            
            n_var = batch.ntvars[ind]
            pre_sols = model_output[index_arrow:index_arrow + n_var].squeeze()[b_vars]
            index_arrow = index_arrow + n_var
            
            pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols == 1).float()
            neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols == 0).float()
            sum_loss = pos_loss + neg_loss
            
            sample_loss = sum_loss * weight[:, None]
            total_sample_loss = sample_loss.sum()
            
            loss_list.append(total_sample_loss)
        
        batch_loss = torch.stack(loss_list)

        return batch_loss, {}