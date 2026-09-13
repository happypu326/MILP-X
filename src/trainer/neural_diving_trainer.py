"""Trainer for the Neural Diving predictor (prediction + SelectiveNet gate)."""

import torch
from .base_trainer import BaseTrainer


class NeuralDivingTrainer(BaseTrainer):
    def _run(self, data_loader, train):
        self.model.train() if train else self.model.eval()
        mean_loss, n = 0.0, 0
        computer = self.train_loss_computer if train else self.val_loss_computer
        for batch in data_loader:
            batch = batch.to(self.device)
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10
            batch_indices = []
            for i in range(batch.nsols.shape[0]):
                batch_indices.extend([i] * len(batch.varInds[i][0][0]))
            batch_indices = torch.tensor(batch_indices, device=self.device)

            out = self.model(
                batch.constraint_features, batch.edge_index, batch.edge_attr,
                batch.variable_features, batch_indices, is_training=train,
            )  # (pred_logit, sel_logit)

            loss, _ = computer.compute(out, batch, is_training=train)
            loss = loss.sum()

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mean_loss += loss.item()
            n += batch.num_graphs
        return mean_loss / max(n, 1)

    def train_epoch(self, data_loader):
        return self._run(data_loader, True)

    @torch.no_grad()
    def validate_epoch(self, data_loader):
        return self._run(data_loader, False)
