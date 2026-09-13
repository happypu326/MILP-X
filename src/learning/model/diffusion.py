"""
Discrete (Bernoulli / 2-state D3PM) diffusion for MILP binary-variable prediction.

Adapts the graph diffusion solver DIFUSCO (Sun & Yang, NeurIPS 2023) from graph
CO (TSP/MIS) to the MILP constraint-variable bipartite graph. There is no
canonical "diffusion-for-MILP" paper; this follows the DIFUSCO recipe:

  * x_0 in {0,1}^{n} is the binary-variable assignment (a solution).
  * Forward: a symmetric 2-state discrete diffusion resamples each bit toward a
    coin flip. With gamma_t = prod_{s<=t}(1 - beta_s), the marginal is
        P(x_t = x_0) = (1 + gamma_t) / 2,   flip prob = (1 - gamma_t) / 2.
  * Reverse: a bipartite GNN denoiser predicts the clean x_0 from (graph, x_t, t).
    The evaluator samples ancestrally (predict x_0, then re-noise to the previous
    timestep via q_sample), which supports arbitrary step striding; the exact
    2-state posterior q(x_{t-1} | x_t, x_0_hat) is also provided
    (`DiscreteBernoulliDiffusion.posterior_prob1`) for full-chain sampling.
  * Training: cross-entropy on the x_0 prediction at random t (variational bound
    surrogate), with the target solution drawn from the (energy-weighted) pool.

Unlike the factorized-Bernoulli PS/Neural-Diving predictors, reverse diffusion
produces correlated, multi-modal samples over the whole binary vector; each
sampled assignment (or the averaged marginal) feeds the usual trust-region
search at test time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import BipartiteGraphConvolution


# --------------------------------------------------------------------------- #
#  Noise schedule + forward/posterior math (symmetric 2-state channel)
# --------------------------------------------------------------------------- #
class DiscreteBernoulliDiffusion:
    def __init__(self, num_timesteps=200, schedule="cosine", device="cpu"):
        self.T = num_timesteps
        self.device = device
        betas = self._make_betas(num_timesteps, schedule)          # [T] (index 1..T -> 0..T-1)
        self.betas = betas.to(device)
        # gamma_t = prod_{s<=t} (1 - beta_s); prepend gamma_0 = 1
        alphas = 1.0 - self.betas
        gamma = torch.cumprod(alphas, dim=0)
        self.gammas = torch.cat([torch.ones(1, device=device), gamma], dim=0)  # [T+1]

    @staticmethod
    def _make_betas(T, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 0.5, T)
        # cosine (Nichol & Dhariwal) mapped to per-step betas, clamped
        steps = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        abar = f / f[0]
        betas = 1 - (abar[1:] / abar[:-1])
        return betas.clamp(1e-4, 0.5)

    def q_sample(self, x0, t):
        """Sample x_t ~ q(x_t | x_0). x0: {0,1} float [N]; t: long [N] in 1..T."""
        gamma_t = self.gammas[t]                     # [N]
        flip_prob = (1.0 - gamma_t) / 2.0
        flip = (torch.rand_like(x0) < flip_prob).float()
        return (x0 * (1 - flip) + (1 - x0) * flip)   # x0 XOR flip

    def posterior_prob1(self, x_t, x0_prob1, t):
        """
        P(x_{t-1} = 1 | x_t, x0_hat) marginalized over the predicted x0
        distribution, per bit. Exact for the symmetric 2-state channel.
        x_t: {0,1} float [N]; x0_prob1: P(x0=1) in [0,1] [N]; t: long scalar/[N].
        """
        b_t = self.betas[t - 1]                       # single-step beta for step t
        g_tm1 = self.gammas[t - 1]                    # cumulative up to t-1
        same_step = 1.0 - b_t / 2.0                   # q(x_t | x_{t-1}=x_t)
        flip_step = b_t / 2.0
        same_cum = (1.0 + g_tm1) / 2.0                # q(x_{t-1}=x0 | x0)
        flip_cum = (1.0 - g_tm1) / 2.0

        # likelihood of the observed x_t given x_{t-1} = 1 or = 0
        lik_xt_given_1 = torch.where(x_t > 0.5, same_step, flip_step)   # x_{t-1}=1
        lik_xt_given_0 = torch.where(x_t < 0.5, same_step, flip_step)   # x_{t-1}=0

        def post_given_x0(x0_val):
            # q(x_{t-1}=1 | x0) and q(x_{t-1}=0 | x0)
            q1 = same_cum if x0_val == 1 else flip_cum
            q0 = flip_cum if x0_val == 1 else same_cum
            u1 = lik_xt_given_1 * q1
            u0 = lik_xt_given_0 * q0
            return u1 / (u1 + u0 + 1e-12)

        p_if_1 = post_given_x0(1)
        p_if_0 = post_given_x0(0)
        return x0_prob1 * p_if_1 + (1 - x0_prob1) * p_if_0


# --------------------------------------------------------------------------- #
#  Sinusoidal timestep embedding
# --------------------------------------------------------------------------- #
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, t):
        # t: float tensor [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


# --------------------------------------------------------------------------- #
#  Bipartite GNN denoiser: predicts x_0 logits from (graph, x_t, t)
# --------------------------------------------------------------------------- #
class DiffusionGNNPolicy(nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        # variable features get one extra channel for the noised state x_t
        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(constraint_nfeats), nn.Linear(constraint_nfeats, emb_size),
            nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(nn.LayerNorm(edge_nfeats))
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(variable_nfeats + 1), nn.Linear(variable_nfeats + 1, emb_size),
            nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU(),
        )
        self.time_embedding = TimeEmbedding(emb_size)

        self.v_to_c = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size), nn.ReLU(), nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, x_t, t, batch_indices=None):
        """
        x_t: current noised binary state, float [n_var] (0 for non-binary vars).
        t:   timestep, long [num_graphs] (or scalar broadcast).
        Returns x_0 logits per variable [n_var].
        """
        n_var = variable_features.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(n_var, dtype=torch.long, device=variable_features.device)

        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        v_in = torch.cat([variable_features, x_t.reshape(-1, 1)], dim=-1)
        c = self.cons_embedding(constraint_features)
        e = self.edge_embedding(edge_features)
        v = self.var_embedding(v_in)

        # add timestep embedding (per graph, broadcast to that graph's variables)
        if t.dim() == 0:
            t = t.reshape(1)
        t_emb = self.time_embedding(t)               # [num_graphs, emb]
        v = v + t_emb[batch_indices]

        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            v_hist.append(v)
        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)
