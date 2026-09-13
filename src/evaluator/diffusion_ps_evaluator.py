"""
Predict-and-Search evaluator whose predictor is the discrete diffusion model.

Reverse-diffuses to a per-binary marginal (averaged over several sampling
chains), then reuses PSFamilyEvaluator's trust-region search verbatim, so the
diffusion predictor is benchmarked under the identical search protocol as PS.
"""

import torch
from typing import List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features, get_raw_constraints
from src.learning.model.diffusion import DiscreteBernoulliDiffusion


class DiffusionPSEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.num_timesteps = config.get("num_timesteps", 200)
        self.inference_steps = config.get("inference_steps", 50)
        self.n_samples = config.get("n_samples", 4)
        self.schedule = config.get("schedule", "cosine")
        # CGD (2608.13079): training-free feasibility projection in reverse process
        self.feasibility_projection = config.get("feasibility_projection", False)
        self.proj_steps = config.get("proj_steps", 5)
        self.proj_lr = config.get("proj_lr", 0.1)
        self.diffusion = DiscreteBernoulliDiffusion(
            num_timesteps=self.num_timesteps, schedule=self.schedule, device=self.device
        )

    def _feasibility_project(self, x0_prob, A_bin, b, sense):
        """
        Constraint-Aware Diffusion (CGD) projection: nudge the predicted binary
        marginals toward the feasible set of A_bin x (<=,>=,==) b by gradient
        descent on the total squared constraint violation. Training-free.

        A_bin: sparse [ncons, |B|] (binary columns only); x0_prob: [|B|].
        """
        if A_bin is None or A_bin._nnz() == 0:
            return x0_prob
        x = x0_prob.clone()
        for _ in range(self.proj_steps):
            Ax = torch.sparse.mm(A_bin, x.unsqueeze(1)).squeeze(1)   # [ncons]
            resid = Ax - b
            # violation direction per constraint
            viol = torch.zeros_like(resid)
            le = (sense == 0); ge = (sense == 1); eq = (sense == 2)
            viol[le] = torch.clamp(resid[le], min=0)
            viol[ge] = torch.clamp(-resid[ge], min=0) * (-1)         # want Ax>=b
            viol[eq] = resid[eq]
            if float(viol.abs().sum()) == 0:
                break
            grad = torch.sparse.mm(A_bin.t(), viol.unsqueeze(1)).squeeze(1)  # [|B|]
            x = (x - self.proj_lr * grad).clamp(0, 1)
        return x

    @torch.no_grad()
    def _sample_marginals(self, cons_f, edge_idx, edge_feat, var_f, b_vars,
                          A_bin=None, b=None, sense=None):
        """Ancestral reverse diffusion (predict-x0 then re-noise), averaged over
        n_samples chains. Returns P(x0=1) for each binary variable [|B|]."""
        n_total = var_f.shape[0]
        n_bin = b_vars.shape[0]
        device = self.device
        b_vars = b_vars.to(device).long()

        # strided descending schedule T -> 1
        steps = torch.linspace(self.num_timesteps, 1, self.inference_steps,
                               device=device).round().long()

        acc = torch.zeros(n_bin, device=device)
        for _ in range(self.n_samples):
            x_bin = (torch.rand(n_bin, device=device) < 0.5).float()
            final_prob = None
            for k in range(steps.shape[0]):
                t = steps[k]
                x_t_full = torch.zeros(n_total, device=device)
                x_t_full[b_vars] = x_bin
                logits = self.model(
                    cons_f.to(device), edge_idx.to(device), edge_feat.to(device),
                    var_f.to(device), x_t_full, t.reshape(1),
                    torch.zeros(n_total, dtype=torch.long, device=device),
                )
                x0_prob = logits.sigmoid()[b_vars]
                if self.feasibility_projection and A_bin is not None:
                    x0_prob = self._feasibility_project(x0_prob, A_bin, b, sense)
                final_prob = x0_prob
                if k < steps.shape[0] - 1:
                    x0_hat = (torch.rand_like(x0_prob) < x0_prob).float()
                    t_prev = steps[k + 1].expand(n_bin)
                    x_bin = self.diffusion.q_sample(x0_hat, t_prev)
            acc += final_prob
        return acc / self.n_samples

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)

        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices, A._values(), use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0],
        )

        A_bin = b = sense = None
        if self.feasibility_projection:
            A_raw, b_raw, sense_raw, _, bcols = get_raw_constraints(ins_path)
            # restrict columns to binary variables (assume other vars contribute 0)
            A_raw = A_raw.coalesce()
            idx, val = A_raw._indices(), A_raw._values()
            col_remap = -torch.ones(A_raw.shape[1], dtype=torch.long)
            col_remap[bcols] = torch.arange(bcols.numel())
            keep = col_remap[idx[1]] >= 0
            A_bin = torch.sparse_coo_tensor(
                torch.stack([idx[0][keep], col_remap[idx[1]][keep]]),
                val[keep], (A_raw.shape[0], bcols.numel())
            ).coalesce().to(self.device)
            b = b_raw.to(self.device); sense = sense_raw.to(self.device)

        marg = self._sample_marginals(
            constraint_features, edge_indices, edge_features, v_nodes, b_vars,
            A_bin=A_bin, b=b, sense=sense,
        ).cpu()

        # scatter binary marginals back to a full per-variable vector
        BD = torch.full((v_nodes.shape[0],), 0.5)
        BD[b_vars.long()] = marg

        all_varname = list(v_map.keys())
        binary_name = {all_varname[i] for i in b_vars.tolist()}
        scores = []
        for i in range(len(v_map)):
            vtype = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], BD[i].item(), -1, vtype])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores
