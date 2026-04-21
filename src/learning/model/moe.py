from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .gcn import GNNEncoder, GNNDecoder
import os
from datetime import datetime


class Expert(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )
        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.mlp(x) + self.residual(x)
        return self.mlp(x)


class GatingNetwork(nn.Module):
    def __init__(self, dim: int, num_dedicate_experts: int,
                 top_k: int, temperature: float = 0.5, bias_lr: float = 1e-3):
        super().__init__()
        self.num_experts = num_dedicate_experts
        self.top_k = top_k
        self.temperature = temperature
        self.bias_lr = bias_lr

        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(dim // 2, num_dedicate_experts),
        )
        # self.gate_proj = nn.Linear(dim, num_dedicate_experts)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.expert_biases = nn.Parameter(torch.zeros(num_dedicate_experts))

    def forward(self, g_emb: torch.Tensor) -> torch.Tensor:
        base_logits = self.gate_proj(g_emb) * self.alpha / self.temperature
        logits = base_logits + self.expert_biases.unsqueeze(0)

        # hard top-k
        _, topk_idx = logits.topk(self.top_k, dim=-1)
       
        mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)

        weights = logits.softmax(dim=-1) * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        return weights

    def update_biases(self, route_weights: torch.Tensor):
        if not self.training:
            return

        expert_load = route_weights.detach().mean(dim=0)
        ideal = expert_load.mean()
        delta = expert_load - ideal

        update = -self.bias_lr * torch.sign(delta)
        self.expert_biases.data.add_(update).clamp_(-1.5, 1.5)


class MoEPolicy(nn.Module):
    def __init__(
        self,
        emb_size: int = 64,
        constraint_nfeats: int = 4,
        edge_nfeats: int = 1,
        variable_nfeats: int = 6,
        num_shared_experts: int = 2,
        num_dedicate_experts: int = 16,
        top_k: int = 4,
        gate_temperature: float = 0.6,
        bias_lr: float = 1e-3,
        dropout: float = 0.1,
        use_dro: bool = True,
        eps_wasserstein: float = 0.2,
        dro_perturb_type: str = "gaussian"
    ):
        super().__init__()
        self.dim = emb_size
        self.Ne = num_dedicate_experts
        self.Ks = num_shared_experts
        self.top_k = top_k

        self.use_dro = use_dro
        self.eps = eps_wasserstein
        self.dro_perturb_type = dro_perturb_type

        self.encoder = GNNEncoder(
            emb_size, constraint_nfeats, edge_nfeats, variable_nfeats
        )

        self.shared_experts = nn.ModuleList(
            [Expert(emb_size) for _ in range(self.Ks)]
        )
        self.dedicate_experts = nn.ModuleList(
            [Expert(emb_size) for _ in range(self.Ne)]
        )

        self.gate = GatingNetwork(
            emb_size,
            num_dedicate_experts,
            top_k,
            temperature=gate_temperature,
            bias_lr=bias_lr,
        )

        self.task_head = GNNDecoder(emb_size)

    def _shared_experts_forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [exp(x) for exp in self.shared_experts]
        return torch.stack(outs, dim=0).mean(0)

    def _dedicate_experts_forward(self, x, route_weights, batch_idx):
        expert_out = torch.stack(
            [exp(x) for exp in self.dedicate_experts], dim=0
        ).permute(1, 0, 2)  # [N, Ne, d]

        # [B, Ne] → [N, Ne]
        batch_weights = route_weights[batch_idx]  # [N, Ne]

        # einsum: (N, Ne) × (N, Ne, d) → (N, d)
        fused = torch.einsum("ne,ned->nd", batch_weights, expert_out)
        return fused

    def _generate_wasserstein_perturbation(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        if self.dro_perturb_type == "gaussian":
            delta = torch.randn_like(z)
        else:
            delta = torch.randn_like(z)
            delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-12)
            delta *= torch.rand(B, 1, device=z.device)

        norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * (self.eps / (norm + 1e-12))
        return delta

    def _dro_robust_loss(
        self,
        gate_input: torch.Tensor,
        route_weights: torch.Tensor,
        v_emb: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_dro or self.eps <= 0:
            return torch.tensor(0.0, device=v_emb.device)

        delta = self._generate_wasserstein_perturbation(gate_input)
        z_tilde = (gate_input + delta).detach()

        adv_route = self.gate(z_tilde)
        adv_fused = self._dedicate_experts_forward(v_emb, adv_route, batch_idx)
        orig_fused = self._dedicate_experts_forward(v_emb, route_weights, batch_idx)
        return F.mse_loss(adv_fused, orig_fused)

    def forward(
        self,
        c_feat: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_attr: torch.Tensor,
        v_feat: torch.Tensor,
        batch_idx: torch.Tensor,
        is_training: bool = False,
    ):
        v_emb, c_emb = self.encoder(c_feat, edge_idx, edge_attr, v_feat, batch_idx)

        g_emb = global_mean_pool(v_emb, batch_idx)  

        route_weights = self.gate(g_emb)                   

        dedicate_fused = self._dedicate_experts_forward(
            v_emb, route_weights, batch_idx
        )  
        shared_fused = self._shared_experts_forward(v_emb)      
        
        combined = v_emb + shared_fused + dedicate_fused        

        logits = self.task_head(combined).squeeze(-1)           

        if is_training:
            self.gate.update_biases(route_weights)

            aux_loss = self._dro_robust_loss(
                g_emb, route_weights, v_emb, batch_idx
            )
            return logits, aux_loss

        return logits, None
