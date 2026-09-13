"""Trainer for guided latent diffusion IP solution generator."""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from src.learning.model.guided_diffusion import GaussianLatentDiffusion
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM


class GuidedDiffusionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = GaussianLatentDiffusion(
            num_timesteps=self.config.get('num_timesteps', 200), device=self.device)
        self.lambda_recon = self.config.get('lambda_recon', 1.0)
        self.lambda_align = self.config.get('lambda_align', 0.1)

    def _step(self, batch, train):
        batch = batch.to(self.device)
        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
        c, ei, ef, v = (batch.constraint_features, batch.edge_index,
                        batch.edge_attr, batch.variable_features)
        solInd = batch.nsols

        # per-graph unpack + sample one target solution per graph
        target_sols, target_vals = [], []
        solEndInd = valEndInd = 0
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])
            solStartInd, solEndInd = solEndInd, solInd[i] * nvar + solEndInd
            valStartInd, valEndInd = valEndInd, valEndInd + solInd[i]
            target_sols.append(batch.solutions[solStartInd:solEndInd].reshape(-1, nvar))
            target_vals.append(batch.objVals[valStartInd:valEndInd])

        n_total = v.shape[0]
        x_full = torch.zeros(n_total, device=self.device)
        bin_mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        batch_idx = torch.zeros(n_total, dtype=torch.long, device=self.device)
        index_arrow = 0
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            wn = ENERGY_WEIGHT_NORM.get(GROUP_CLASS[batch.group[ind].cpu().item()], 1)
            varInds = batch.varInds[ind]
            varname_map, b_vars = varInds[0][0], varInds[1][0].long()
            sols_b = sols[:, varname_map][:, b_vars]
            w = torch.softmax(-vals / wn, dim=0)
            x0 = sols_b[torch.multinomial(w, 1).item()].float()
            n_var = int(batch.ntvars[ind])
            cols = b_vars + index_arrow
            x_full[cols] = x0
            bin_mask[cols] = True
            batch_idx[index_arrow:index_arrow + n_var] = ind
            index_arrow += n_var

        h_v, z_i = self.model.encode_instance(c, ei, ef, v, batch_idx)
        z_s = self.model.encode_solution(c, ei, ef, v, x_full, batch_idx)

        # reconstruction
        recon_logits = self.model.decode(h_v, z_s, batch_idx)
        if bin_mask.any():
            L_recon = F.binary_cross_entropy_with_logits(recon_logits[bin_mask], x_full[bin_mask])
        else:
            L_recon = recon_logits.sum() * 0.0

        # latent diffusion (eps prediction)
        num_graphs = z_i.shape[0]
        t = torch.randint(0, self.diffusion.T, (num_graphs,), device=self.device)
        noise = torch.randn_like(z_s)
        z_t = self.diffusion.q_sample(z_s, t, noise)
        eps_pred = self.model.predict_eps(z_t, t, z_i)
        L_diff = F.mse_loss(eps_pred, noise)

        # instance/solution latent alignment
        L_align = F.mse_loss(z_s, z_i)

        return L_diff + self.lambda_recon * L_recon + self.lambda_align * L_align

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
