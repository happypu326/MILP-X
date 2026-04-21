import torch
import numpy as np
from typing import Tuple

from src.learning.loss.base_loss import BaseLossComputer
from src.learning.loss.bce_loss import BCELossComputer


class RoMELossComputer(BaseLossComputer):
    def __init__(self, config: dict):
        super().__init__(config)

        self.bce_loss = BCELossComputer(config)

        self.device = config.get('device', 'cpu')
        self.is_robust = config.get('is_robust', True)
        self.gamma = config.get('gamma', 0.1)
        self.alpha = config.get('alpha', 0.2)
        self.min_var_weight = config.get('min_var_weight', 0)
        self.step_size = config.get('step_size', 0.01)
        self.normalize_loss = config.get('normalize_loss', False)
        self.other_loss_ratio = config.get('other_loss_ratio', 0.1)
        self.btl = config.get('btl', False)

        group_stats = config.get('group_stats', None)
        if group_stats is not None:
            self.n_groups = group_stats['n_groups']
            self.group_counts = group_stats['group_counts'].float().to(self.device)
            self.group_frac = group_stats['group_frac'].float().to(self.device)
        else:
            self.n_groups = 1
            self.group_counts = torch.ones(1, device=self.device)
            self.group_frac = torch.ones(1, device=self.device)

        adj = config.get('adj', None)
        if adj is None:
            self.adj = torch.zeros(self.n_groups, device=self.device)
        else:
            self.adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
        
        if self.is_robust:
            assert self.alpha, 'alpha must be specified'
        
        self.adv_probs = torch.ones(self.n_groups, device=self.device) / self.n_groups

        self.exp_avg_loss = torch.zeros(self.n_groups, device=self.device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, dtype=torch.bool, device=self.device)

        self.reset_stats()

    def compute(self, model_output, batch, is_training=False) -> Tuple[torch.Tensor, dict]:
        batch_loss, bce_info = self.bce_loss.compute(model_output, batch, is_training)

        group = batch.group
        loss = self.dro_loss(batch_loss, group, is_training=is_training)

        info = {
            'batch_loss': batch_loss.detach(),
            'dro_loss': loss.detach(),
            'adv_probs': self.adv_probs.detach().clone(),
            **bce_info,
        }

        return loss, info

    def dro_loss(self, per_sample_losses, group_idx=None, is_training=True):
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        self.update_exp_avg_loss(group_loss, group_count)

        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        self.update_stats(actual_loss, group_loss, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)
    
    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def update_stats(self, actual_loss, group_loss, group_count, weights=None):
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss

        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups, device=self.device)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.batch_count = 0