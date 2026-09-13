"""
Trainer for the discrete Bernoulli diffusion MILP solution predictor.
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from src.learning.model.diffusion import DiscreteBernoulliDiffusion
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM


class DiffusionTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, val_dataloader,
                 train_loss_computer, val_loss_computer, config):
        super().__init__(model, train_dataloader, val_dataloader,
                         train_loss_computer, val_loss_computer, config)
        self.diffusion = DiscreteBernoulliDiffusion(
            num_timesteps=config.get('num_timesteps', 200),
            schedule=config.get('schedule', 'cosine'),
            device=self.device,
        )

    def _step(self, batch, is_training):
        batch = batch.to(self.device)
        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

        solInd = batch.nsols
        # unpack per-graph solution pools (same layout as BCELossComputer)
        target_sols, target_vals = [], []
        solEndInd = valEndInd = 0
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])
            solStartInd, solEndInd = solEndInd, solInd[i] * nvar + solEndInd
            valStartInd, valEndInd = valEndInd, valEndInd + solInd[i]
            target_sols.append(batch.solutions[solStartInd:solEndInd].reshape(-1, nvar))
            target_vals.append(batch.objVals[valStartInd:valEndInd])

        n_total = batch.variable_features.shape[0]
        x_t_full = torch.zeros(n_total, device=self.device)
        x0_full = torch.zeros(n_total, device=self.device)
        bin_mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        t_per_graph = torch.randint(1, self.diffusion.T + 1, (solInd.shape[0],), device=self.device)
        batch_indices = torch.zeros(n_total, dtype=torch.long, device=self.device)

        index_arrow = 0
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            group_ind = batch.group[ind].cpu().item()
            weight_norm = ENERGY_WEIGHT_NORM.get(GROUP_CLASS[group_ind], 1)
            varInds = batch.varInds[ind]
            varname_map = varInds[0][0]
            b_vars = varInds[1][0].long()
            sols_b = sols[:, varname_map][:, b_vars]              # [S, |B|]

            # sample one target solution ~ Boltzmann weight
            w = torch.softmax(-vals / weight_norm, dim=0)
            pick = torch.multinomial(w, 1).item()
            x0_bin = sols_b[pick].float()                        # [|B|]

            n_var = int(batch.ntvars[ind])
            local_bidx = b_vars + index_arrow                    # positions in concat order
            x0_full[local_bidx] = x0_bin
            bin_mask[local_bidx] = True
            t_bin = t_per_graph[ind].expand(x0_bin.shape[0])
            x_t_full[local_bidx] = self.diffusion.q_sample(x0_bin, t_bin)
            batch_indices[index_arrow:index_arrow + n_var] = ind
            index_arrow += n_var

        logits = self.model(
            batch.constraint_features, batch.edge_index, batch.edge_attr,
            batch.variable_features, x_t_full, t_per_graph, batch_indices,
        )
        # cross-entropy on x0 prediction over binary positions only
        if bin_mask.sum() == 0:
            return logits.sum() * 0.0
        loss = F.binary_cross_entropy_with_logits(
            logits[bin_mask], x0_full[bin_mask]
        )
        return loss

    def train_epoch(self, data_loader):
        self.model.train()
        mean_loss, n = 0.0, 0
        for batch in data_loader:
            loss = self._step(batch, True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_loss += loss.item()
            n += batch.num_graphs
        return mean_loss / max(n, 1)

    @torch.no_grad()
    def validate_epoch(self, data_loader):
        self.model.eval()
        mean_loss, n = 0.0, 0
        for batch in data_loader:
            loss = self._step(batch, False)
            mean_loss += loss.item()
            n += batch.num_graphs
        return mean_loss / max(n, 1)
