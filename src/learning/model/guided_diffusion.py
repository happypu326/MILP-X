"""
Guided latent diffusion for feasible IP solution generation.

Song et al. (approach of arXiv 2406.12349, "Effective Generation of Feasible
Solutions for Integer Programming via Guided Diffusion").

Pipeline (all conditioned on the constraint-variable bipartite graph):
  * instance encoder  E_i : graph            -> per-variable embeddings h_v and
                                                a graph-level instance latent z_i
  * solution encoder  E_s : graph + solution -> solution latent z_s
  * latent denoiser        : Gaussian DDPM on z_s conditioned on z_i (predicts eps)
  * decoder           D    : (h_v, z) -> per-variable Bernoulli logits

Training (joint): reconstruction BCE(D(z_s), x) + latent-diffusion MSE(eps) +
an alignment term ||mean_k z_s^k - z_i||^2 that pulls the instance latent toward
its solutions' latents (a batch-size-1-friendly proxy for the paper's
instance/solution contrastive alignment).

Inference: sample z ~ N(0,I), reverse-diffuse conditioned on z_i to z_hat,
decode to per-binary marginals; an optional feasibility guidance step steers the
decoded marginals toward the feasible set. The marginals then drive the usual
trust-region search.
"""

import math
import torch
import torch.nn as nn

from .gcn import GNNEncoder
from .diffusion import TimeEmbedding
from torch_geometric.nn import global_mean_pool


class GaussianLatentDiffusion:
    def __init__(self, num_timesteps=200, device="cpu"):
        self.T = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)          # [T]

    def q_sample(self, z0, t, noise):
        acp_t = self.acp[t].unsqueeze(-1)
        return acp_t.sqrt() * z0 + (1 - acp_t).sqrt() * noise

    @torch.no_grad()
    def ddpm_step(self, z_t, eps, t):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        acp_t = self.acp[t]
        coef = (1 - alpha_t) / (1 - acp_t).sqrt()
        mean = (z_t - coef * eps) / alpha_t.sqrt()
        if t > 0:
            noise = torch.randn_like(z_t)
            return mean + beta_t.sqrt() * noise
        return mean


class GuidedDiffusionModel(nn.Module):
    def __init__(self, emb_size=64, latent_dim=64, constraint_nfeats=4,
                 edge_nfeats=2, variable_nfeats=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.instance_encoder = GNNEncoder(emb_size, constraint_nfeats, edge_nfeats, variable_nfeats)
        self.solution_encoder = GNNEncoder(emb_size, constraint_nfeats, edge_nfeats, variable_nfeats + 1)
        self.inst_proj = nn.Linear(emb_size, latent_dim)
        self.sol_proj = nn.Linear(emb_size, latent_dim)
        self.time_emb = TimeEmbedding(latent_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim * 2 + latent_dim, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_size + latent_dim, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

    def encode_instance(self, c, ei, ef, v, batch):
        enc = self.instance_encoder(c, ei, ef, v, batch_indices=batch)
        h_v = enc[0] if isinstance(enc, tuple) else enc
        z_i = self.inst_proj(global_mean_pool(h_v, batch))
        return h_v, z_i

    def encode_solution(self, c, ei, ef, v, x_full, batch):
        v_aug = torch.cat([v, x_full.reshape(-1, 1)], dim=-1)
        enc = self.solution_encoder(c, ei, ef, v_aug, batch_indices=batch)
        h = enc[0] if isinstance(enc, tuple) else enc
        return self.sol_proj(global_mean_pool(h, batch))

    def predict_eps(self, z_t, t, z_i):
        te = self.time_emb(t.float())
        return self.denoiser(torch.cat([z_t, te, z_i], dim=-1))

    def decode(self, h_v, z, batch):
        z_b = z[batch]                                   # broadcast latent to vars
        return self.decoder(torch.cat([h_v, z_b], dim=-1)).squeeze(-1)
