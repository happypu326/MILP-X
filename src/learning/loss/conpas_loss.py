"""
ConPaS: Contrastive Predict-and-Search loss.

Huang, Ferber, Zharmagambetov, Tian, Dilkina. "Contrastive Predict-and-Search
for Mixed Integer Linear Programs." ICML 2024.

Vanilla PS regresses the predicted marginals onto an energy-weighted average of
the solution pool with independent per-variable BCE. ConPaS instead trains the
predictor *contrastively*: it pulls the predicted distribution toward
high-quality (near-optimal) solutions and pushes it away from poor ones, so the
model learns a decision boundary between good and bad solutions rather than an
average.

Given the predicted marginals p_j in (0,1), the log-likelihood score of a
candidate assignment s under the factorized Bernoulli is

    phi(s) = (1/|B|) * sum_{j in B} [ s_j log p_j + (1 - s_j) log(1 - p_j) ]

(averaged over binaries for scale-invariance). With a positive set S+
(best-objective solutions) and a negative set S- (worst-objective solutions
from the pool), the InfoNCE objective per instance is

    L = - (1/|S+|) sum_{s+ in S+} log
            exp(phi(s+)/tau) / ( sum_{s in S+ U S-} exp(phi(s)/tau) )

which is computed stably with logsumexp.

Drop-in: same bipartite GNN backbone and the same trust-region search at test
time; only the training objective changes.
"""

import torch
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM


class ConPaSLossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.tau = config.get('tau', 0.1)
        # fraction of the (weight-ranked) pool treated as positives / negatives
        self.pos_frac = config.get('pos_frac', 0.2)
        self.neg_frac = config.get('neg_frac', 0.5)
        self.eps = 1e-6

    def _phi(self, probs, sols):
        """Mean per-binary log-likelihood of each solution. sols: [S, |B|],
        probs: [|B|]. Returns [S]."""
        p = probs.clamp(self.eps, 1 - self.eps)
        logp = torch.log(p)[None, :]
        log1p = torch.log(1 - p)[None, :]
        ll = sols * logp + (1 - sols) * log1p          # [S, |B|]
        return ll.mean(dim=1)                          # [S]

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        group = batch.group
        solInd = batch.nsols

        # unpack the per-graph solution pools (same layout as BCELossComputer)
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
            sols_b = sols[:, varname_map][:, b_vars]                       # [S, |B|]

            n_var = batch.ntvars[ind]
            probs = model_output[index_arrow:index_arrow + n_var].squeeze()[b_vars]
            index_arrow += n_var

            if b_vars.numel() == 0:
                continue

            S = sols_b.shape[0]
            if S < 2:            # need at least one positive and one negative
                continue

            # rank solutions by Boltzmann weight (higher = better objective)
            w = torch.exp(-vals / weight_norm)
            order = torch.argsort(w, descending=True)                     # best first
            n_pos = max(1, int(round(self.pos_frac * S)))
            n_neg = max(1, int(round(self.neg_frac * S)))
            n_pos = min(n_pos, S - 1)
            pos_idx = order[:n_pos]
            neg_idx = order[-n_neg:]
            # ensure disjoint (drop overlap from negatives)
            pos_set = set(pos_idx.tolist())
            neg_idx = torch.tensor([j for j in neg_idx.tolist() if j not in pos_set],
                                   device=sols_b.device, dtype=torch.long)
            if neg_idx.numel() == 0:
                continue

            phi = self._phi(probs, sols_b) / self.tau                     # [S]
            phi_pos = phi[pos_idx]                                        # [P]
            phi_neg = phi[neg_idx]                                        # [N]

            # InfoNCE per positive: -(phi_pos - logsumexp([phi_pos, phi_neg...]))
            neg_lse = torch.logsumexp(phi_neg, dim=0)                     # scalar
            denom = torch.logaddexp(phi_pos, neg_lse.expand_as(phi_pos))  # [P]
            per_pos = -(phi_pos - denom)
            loss_list.append(per_pos.mean())

        if not loss_list:
            zero = model_output.sum() * 0.0
            return zero.reshape(1), {}

        batch_loss = torch.stack(loss_list)
        return batch_loss, {'conpas_loss': batch_loss.detach().mean().item()}
