"""
Trainer for SHSP conditional predictor.

For each instance: sample a target solution ~ Boltzmann weight, randomly reveal
a fraction rho ~ U(0, rho_max) of its binaries as conditioning (known_mask=1,
known_value=true), and train the model to predict the *unrevealed* binaries with
weighted BCE. This teaches the conditional distribution used by hierarchical
decoding at test time.
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM


class SHSPTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_max = self.config.get('reveal_max', 0.8)

    def _step(self, batch, train):
        batch = batch.to(self.device)
        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
        solInd = batch.nsols

        target_sols, target_vals = [], []
        solEndInd = valEndInd = 0
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])
            solStartInd, solEndInd = solEndInd, solInd[i] * nvar + solEndInd
            valStartInd, valEndInd = valEndInd, valEndInd + solInd[i]
            target_sols.append(batch.solutions[solStartInd:solEndInd].reshape(-1, nvar))
            target_vals.append(batch.objVals[valStartInd:valEndInd])

        n_total = batch.variable_features.shape[0]
        known_mask = torch.zeros(n_total, device=self.device)
        known_value = torch.zeros(n_total, device=self.device)
        target = torch.zeros(n_total, device=self.device)
        predict_mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        batch_indices = torch.zeros(n_total, dtype=torch.long, device=self.device)

        index_arrow = 0
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            wn = ENERGY_WEIGHT_NORM.get(GROUP_CLASS[batch.group[ind].cpu().item()], 1)
            varInds = batch.varInds[ind]
            varname_map, b_vars = varInds[0][0], varInds[1][0].long()
            sols_b = sols[:, varname_map][:, b_vars]
            w = torch.softmax(-vals / wn, dim=0)
            x0 = sols_b[torch.multinomial(w, 1).item()].float()      # [|B|]

            n_var = int(batch.ntvars[ind])
            cols = b_vars + index_arrow
            target[cols] = x0
            # random reveal a fraction as known conditioning
            rho = torch.rand(1, device=self.device).item() * self.rho_max
            reveal = torch.rand(x0.shape[0], device=self.device) < rho
            known_mask[cols[reveal]] = 1.0
            known_value[cols[reveal]] = x0[reveal]
            predict_mask[cols[~reveal]] = True
            batch_indices[index_arrow:index_arrow + n_var] = ind
            index_arrow += n_var

        logits = self.model(
            batch.constraint_features, batch.edge_index, batch.edge_attr,
            batch.variable_features, known_mask, known_value, batch_indices,
            is_training=train,
        )
        if predict_mask.sum() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[predict_mask], target[predict_mask])

    def train_epoch(self, data_loader):
        self.model.train()
        tot, n = 0.0, 0
        for batch in data_loader:
            loss = self._step(batch, True)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            tot += loss.item(); n += batch.num_graphs
        return tot / max(n, 1)

    @torch.no_grad()
    def validate_epoch(self, data_loader):
        self.model.eval()
        tot, n = 0.0, 0
        for batch in data_loader:
            tot += self._step(batch, False).item(); n += batch.num_graphs
        return tot / max(n, 1)
