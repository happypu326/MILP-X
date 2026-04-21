"""
PS_family (Predict and Search) 训练器
"""

import torch
from .base_trainer import BaseTrainer
from src.utils.utils import GROUP_CLASS, ENERGY_WEIGHT_NORM

class PS_Family_Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader,
        val_dataloader,
        train_loss_computer,
        val_loss_computer,
        config: dict
    ):
        super().__init__(model, train_dataloader, val_dataloader, train_loss_computer, val_loss_computer, config)
        self.gnn_type = config.get('gnn_type', 'gcn')
        self.other_loss_ratio = config.get('other_loss_ratio', 0.1)
    
    def train_epoch(self, data_loader) -> float:
        self.model.train()
        mean_loss = 0
        n_samples_processed = 0
        
        for step, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10

            batch_indices = []
            solInd = batch.nsols

            for i in range(solInd.shape[0]):#for in batch
                nvar = len(batch.varInds[i][0][0])
                batch_indices.extend([i] * nvar)

            BD = self.model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                torch.tensor(batch_indices, device=self.device),
                is_training = True
            )

            if self.gnn_type == 'moe':
                BD, other_loss = BD

            BD = BD.sigmoid()

            loss, _ = self.train_loss_computer.compute(BD, batch, is_training = True)
            
            if self.method_type == 'RoME':
                final_loss = loss + self.other_loss_ratio * other_loss
            else:
                final_loss = loss.sum()

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
            
            mean_loss += final_loss.item()
            n_samples_processed += batch.num_graphs
        
        mean_loss /= n_samples_processed
        return mean_loss
    
    def validate_epoch(self, data_loader) -> float:
        self.model.eval()
        mean_loss = 0
        n_samples_processed = 0
        
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                batch.constraint_features[torch.isinf(batch.constraint_features)] = 10       

                batch_indices = []
                solInd = batch.nsols

                for i in range(solInd.shape[0]):
                    nvar = len(batch.varInds[i][0][0])
                    batch_indices.extend([i] * nvar)

                BD = self.model(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
                    torch.tensor(batch_indices, device=self.device),
                    is_training = False
                )

                if self.gnn_type == 'moe':
                    BD, other_loss = BD 

                BD = BD.sigmoid()

                loss, _ = self.val_loss_computer.compute(BD, batch, is_training = False)
                final_loss = loss.sum()

                mean_loss += final_loss.item()
                n_samples_processed += batch.num_graphs
        
        mean_loss /= n_samples_processed
        return mean_loss
