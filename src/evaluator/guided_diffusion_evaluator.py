"""
Guided latent-diffusion evaluator (arXiv 2406.12349).

Encodes the instance, samples a solution latent by reverse Gaussian diffusion
conditioned on the instance latent, decodes to per-binary marginals, optionally
applies a feasibility-guidance projection, and feeds the marginals to the usual
trust-region search.
"""

import torch
from typing import List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features, get_raw_constraints
from src.learning.model.guided_diffusion import GaussianLatentDiffusion


class GuidedDiffusionEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.num_timesteps = config.get("num_timesteps", 200)
        self.n_samples = config.get("n_samples", 4)
        self.feasibility_projection = config.get("feasibility_projection", False)
        self.proj_steps = config.get("proj_steps", 5)
        self.proj_lr = config.get("proj_lr", 0.1)
        self.diffusion = GaussianLatentDiffusion(num_timesteps=self.num_timesteps, device=self.device)

    def _project(self, p, A_bin, b, sense):
        if A_bin is None or A_bin._nnz() == 0:
            return p
        x = p.clone()
        for _ in range(self.proj_steps):
            resid = torch.sparse.mm(A_bin, x.unsqueeze(1)).squeeze(1) - b
            viol = torch.zeros_like(resid)
            viol[sense == 0] = torch.clamp(resid[sense == 0], min=0)
            viol[sense == 1] = -torch.clamp(-resid[sense == 1], min=0)
            viol[sense == 2] = resid[sense == 2]
            if float(viol.abs().sum()) == 0:
                break
            grad = torch.sparse.mm(A_bin.t(), viol.unsqueeze(1)).squeeze(1)
            x = (x - self.proj_lr * grad).clamp(0, 1)
        return x

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        device = self.device
        cons_f = c_nodes.cpu(); cons_f[torch.isnan(cons_f)] = 1
        cons_f = cons_f.to(device)
        ei = A._indices().to(device)
        ef = build_edge_features(A._indices(), A._values(),
                                 use_edge_coeff=self.use_edge_coeff,
                                 num_cons=c_nodes.shape[0]).to(device)
        var_f = v_nodes.to(device)
        n_total = v_nodes.shape[0]
        b_idx = b_vars.to(device).long()
        batch = torch.zeros(n_total, dtype=torch.long, device=device)

        h_v, z_i = self.model.encode_instance(cons_f, ei, ef, var_f, batch)

        A_bin = b = sense = None
        if self.feasibility_projection:
            A_raw, b_raw, sense_raw, _, bcols = get_raw_constraints(ins_path)
            A_raw = A_raw.coalesce(); idx, val = A_raw._indices(), A_raw._values()
            remap = -torch.ones(A_raw.shape[1], dtype=torch.long); remap[bcols] = torch.arange(bcols.numel())
            keep = remap[idx[1]] >= 0
            A_bin = torch.sparse_coo_tensor(torch.stack([idx[0][keep], remap[idx[1]][keep]]),
                                            val[keep], (A_raw.shape[0], bcols.numel())).coalesce().to(device)
            b = b_raw.to(device); sense = sense_raw.to(device)

        acc = torch.zeros(b_idx.numel(), device=device)
        for _ in range(self.n_samples):
            z = torch.randn(1, self.model.latent_dim, device=device)
            for t in range(self.diffusion.T - 1, -1, -1):
                eps = self.model.predict_eps(z, torch.tensor([t], device=device), z_i)
                z = self.diffusion.ddpm_step(z, eps, torch.tensor(t, device=device))
            logits = self.model.decode(h_v, z, batch)
            p = logits.sigmoid()[b_idx]
            if self.feasibility_projection and A_bin is not None:
                p = self._project(p, A_bin, b, sense)
            acc += p
        marg = (acc / self.n_samples).cpu()

        BD = torch.full((n_total,), 0.5)
        BD[b_idx.cpu()] = marg
        all_varname = list(v_map.keys())
        binary_name = {all_varname[i] for i in b_vars.tolist()}
        scores = []
        for i in range(len(v_map)):
            vtype = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], BD[i].item(), -1, vtype])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores
