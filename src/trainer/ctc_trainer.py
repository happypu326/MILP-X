"""
Constraint Matters trainer: dual-head (variable + constraint) prediction with a
dual focal loss. Mirrors PS_Family_Trainer but the model returns a
(var_logits, con_logits) tuple, both passed through sigmoid before the loss.
"""

import torch

from .base_trainer import BaseTrainer


class CTC_Trainer(BaseTrainer):
    def _run(self, data_loader, is_training):
        self.model.train() if is_training else self.model.eval()
        mean_loss = 0.0
        n_samples = 0
        loss_computer = self.train_loss_computer if is_training else self.val_loss_computer

        for batch in data_loader:
            batch = batch.to(self.device)
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

            batch_indices = []
            for i in range(batch.nsols.shape[0]):
                nvar = len(batch.varInds[i][0][0])
                batch_indices.extend([i] * nvar)

            var_logit, con_logit = self.model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                torch.tensor(batch_indices, device=self.device),
                is_training=is_training,
            )
            out = (var_logit.sigmoid(), con_logit.sigmoid())
            loss, _ = loss_computer.compute(out, batch, is_training=is_training)
            final_loss = loss.sum()

            if is_training:
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

            mean_loss += final_loss.item()
            n_samples += batch.num_graphs

        return mean_loss / max(n_samples, 1)

    def train_epoch(self, data_loader) -> float:
        return self._run(data_loader, is_training=True)

    @torch.no_grad()
    def validate_epoch(self, data_loader) -> float:
        return self._run(data_loader, is_training=False)
