"""
CL-LNS trainer: trains the destroy policy with the contrastive (InfoNCE) loss
over expert vs perturbed destroy actions. One LNS state per item (batch_size=1);
the model outputs a per-variable destroy score (sigmoid of the encoder logits).
"""

import torch

from .base_trainer import BaseTrainer


class CLLNS_Trainer(BaseTrainer):
    def _run(self, data_loader, is_training):
        self.model.train() if is_training else self.model.eval()
        loss_computer = self.train_loss_computer if is_training else self.val_loss_computer
        mean_loss = 0.0
        n = 0
        for batch in data_loader:
            batch = batch.to(self.device)
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
            n_var = batch.variable_features.shape[0]
            bi = torch.zeros(n_var, dtype=torch.long, device=self.device)

            out = self.model(batch.constraint_features, batch.edge_index, batch.edge_attr,
                             batch.variable_features, bi, is_training=is_training)
            if isinstance(out, tuple):
                out = out[0]
            pi = out.sigmoid()
            loss, _ = loss_computer.compute(pi, batch, is_training=is_training)
            final_loss = loss.sum()

            if is_training:
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
            mean_loss += final_loss.item()
            n += 1
        return mean_loss / max(n, 1)

    def train_epoch(self, data_loader) -> float:
        return self._run(data_loader, is_training=True)

    @torch.no_grad()
    def validate_epoch(self, data_loader) -> float:
        return self._run(data_loader, is_training=False)
