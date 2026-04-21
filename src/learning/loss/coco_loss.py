import torch
import torch.nn.functional as F
from src.utils.utils import ENERGY_WEIGHT_NORM

class CoCoLossComputer:
    def __init__(self, config):
        self.margin = config.get('margin', 0.9)
        self.alpha = config.get('alpha', 0.01)
        self.tao = config.get('tao', 0.1)
        self.weight_norm = config.get('weight_norm', 100)
        self.problem = config.get('problem_type', 'CA')
        self.weight_norm = ENERGY_WEIGHT_NORM.get(self.problem, 1)

    def compute(self, model_output, batch, is_training=False):
        batch_info = []

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

        MSCL_loss = 0.0
        index_arrow = 0 

        for sols, vals, n_var, varInds in zip(
            target_sols,
            target_vals,
            batch.ntvars,
            batch.varInds,
        ):
            exp_weight = torch.exp(-vals / self.weight_norm)
            weight = exp_weight / exp_weight.sum()  

            varname_map = varInds[0][0]
            b_vars = varInds[1][0].long()
            sols = sols[:, varname_map][:, b_vars] 

            pre_sols = model_output[index_arrow : index_arrow + n_var].squeeze()[b_vars]  # (|B|,)
            index_arrow += n_var

            logits = torch.logit(pre_sols.clamp(min=1e-8, max=1 - 1e-8)) / self.tao  # (|B|,)
            exp_logits = torch.exp(logits)
            partition = exp_logits.sum()  

            for sol_vec, w_q in zip(sols, weight):
                pos_mask = sol_vec.bool()
                if pos_mask.sum() == 0:  
                    continue
                numerator = exp_logits[pos_mask].sum()
                mscl_loss = -w_q * torch.log(numerator / partition)
                MSCL_loss += mscl_loss

        Rank_loss = 0
        index_arrow = 0
        
        for _,(sols,vals) in enumerate(zip(target_sols,target_vals)):
            n_vals = vals
            exp_weight = torch.exp(-n_vals/self.weight_norm)
            weight = exp_weight/exp_weight.sum()

            varInds = batch.varInds[_]
            varname_map=varInds[0][0]
            b_vars=varInds[1][0].long()

            sols = sols[:,varname_map][:,b_vars]
            
            n_var = batch.ntvars[_]
            pre_sols = model_output[index_arrow:index_arrow + n_var].squeeze()[b_vars]
            index_arrow = index_arrow + n_var
            
            rank_loss = 0
            for sol_idx in range(sols.size(0)):
                cur_sol = sols[sol_idx]
                pos_mask = cur_sol == 1
                neg_mask = cur_sol == 0
                if not torch.any(pos_mask) or not torch.any(neg_mask):
                    continue
                
                pos_scores = pre_sols[pos_mask]
                neg_scores = pre_sols[neg_mask]
                
                diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                pair_losses = torch.clamp(self.margin - diff, min=0)
                sol_rank_loss = pair_losses.mean()
                rank_loss += weight[sol_idx] * sol_rank_loss
                
            Rank_loss += rank_loss

        Loss = MSCL_loss + self.alpha * Rank_loss

        batch_info.append({
            'mscl_loss': MSCL_loss.item(),
            'rank_loss': Rank_loss.item()
        })
        
        return Loss, batch_info

    def _compute_mscl_loss(self, pre_sols, sols, weight):
        MSCL_loss = 0.0
        logits = torch.logit(pre_sols.clamp(min=1e-8, max=1 - 1e-8)) / self.tao  # (|B|,)
        exp_logits = torch.exp(logits)
        partition = exp_logits.sum()  # scalar Z

        for sol_vec, w_q in zip(sols, weight):
            pos_mask = sol_vec.bool()
            if pos_mask.sum() == 0:  # no positive variables in this solution
                continue
            numerator = exp_logits[pos_mask].sum()
            mscl_loss = -w_q * torch.log(numerator / partition)
            MSCL_loss += mscl_loss

        return MSCL_loss

    def _compute_rank_loss(self, pre_sols, sols, weight):
        rank_loss = 0
        for sol_idx in range(sols.size(0)):
            cur_sol = sols[sol_idx]
            pos_mask = cur_sol == 1
            neg_mask = cur_sol == 0
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue
            
            pos_scores = pre_sols[pos_mask]
            neg_scores = pre_sols[neg_mask]
            
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            pair_losses = torch.clamp(self.margin - diff, min=0)
            sol_rank_loss = pair_losses.mean()
            rank_loss += weight[sol_idx] * sol_rank_loss

        return rank_loss
