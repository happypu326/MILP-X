from .base_trainer import BaseTrainer
import torch

class CoCoTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, train_loss_computer, val_loss_computer, config):
        super().__init__(model, train_dataloader, val_dataloader, train_loss_computer, val_loss_computer, config)

    def train_epoch(self, data_loader):
        self.model.train()
        mean_loss = 0.0
        n_samples_processed = 0

        for _, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            
            constraint_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntcons), device=batch.ntcons.device), batch.ntcons.clone().detach().long())
            variable_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntvars), device=batch.ntvars.device), batch.ntvars.clone().detach().long())

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 

            output = self.model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.n_constraints, 
                constraint_features_batch,
                variable_features_batch,
            ).sigmoid()  

            loss, info = self.train_loss_computer.compute(output, batch, is_training=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs

        return mean_loss / n_samples_processed

    @torch.no_grad()
    def validate_epoch(self, data_loader):
        self.model.eval()
        mean_loss = 0.0
        n_samples_processed = 0

        for _, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            
            constraint_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntcons), device=batch.ntcons.device), batch.ntcons.clone().detach().long())
            variable_features_batch = torch.repeat_interleave(torch.arange(len(batch.ntvars), device=batch.ntvars.device), batch.ntvars.clone().detach().long())

            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # sanitize

            output = self.model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.n_constraints,
                constraint_features_batch,
                variable_features_batch,
            ).sigmoid() 

            loss, info = self.val_loss_computer.compute(output, batch, is_training=False)

            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs

        return mean_loss / n_samples_processed