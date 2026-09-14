"""
CL-LNS destroy-policy loss (Huang et al., "Searching Large Neighborhoods for ILPs
with Contrastive Learning", ICML 2023).

Per LNS state s = (ILP + incumbent), the policy outputs a per-variable destroy
score pi(s) in [0,1]^n. A destroy ACTION a is a binary vector over (binary)
variables (1 = unfix/destroy). Positive actions come from a Local-Branching
expert (the variables LB chose to flip); negative actions are
cardinality-preserving perturbations of that subset (swap which variables are
destroyed, keeping the count). The score of an action is the MEAN destroy score
over its selected variables, phi(a) = (a . pi) / |a| (normalized so the contrast
is about WHICH variables to destroy, not how many), and the policy is trained
with InfoNCE:

    L = - (1/|Sp|) sum_{a+ in Sp}
            log( exp(phi(a+)/tau) / sum_{a in Sn U {a+}} exp(phi(a)/tau) )

so it learns to score the expert's destroy subsets above the perturbed ones.
Computed stably with logsumexp. Assumes batch_size = 1 (one LNS state per item).
"""

import torch
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer


class CLLNSLossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.tau = config.get('tau', 0.07)

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        # per-variable destroy scores over binary variables (graph order)
        b_vars = batch.varInds[0][1][0].long()
        pi = model_output.squeeze()[b_vars]                 # [nb] in [0,1]
        nb = pi.shape[0]

        pos = batch.pos_actions.reshape(-1, nb).to(pi.device)   # [P, nb]
        neg = batch.neg_actions.reshape(-1, nb).to(pi.device)   # [N, nb]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            return (model_output.sum() * 0.0).reshape(1), {}

        # mean destroy score of each action (scale-invariant to action cardinality)
        phi_pos = (pos * pi[None, :]).sum(dim=1) / pos.sum(dim=1).clamp(min=1.0) / self.tau  # [P]
        phi_neg = (neg * pi[None, :]).sum(dim=1) / neg.sum(dim=1).clamp(min=1.0) / self.tau  # [N]
        neg_lse = torch.logsumexp(phi_neg, dim=0)               # scalar
        denom = torch.logaddexp(phi_pos, neg_lse.expand_as(phi_pos))
        loss = -(phi_pos - denom).mean()
        return loss.reshape(1), {'cllns_loss': float(loss.detach())}
