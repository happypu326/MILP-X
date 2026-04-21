from .base_trainer import BaseTrainer
import torch
from torch_geometric.utils import unbatch
from tqdm import tqdm

class DiffILOTrainer(BaseTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, train_loss_computer, val_loss_computer, config):
        super().__init__(model, train_dataloader, val_dataloader, train_loss_computer, val_loss_computer, config)

    def train_epoch(self, data_loader):
        self.model.train()
        num_samples = 0
        epoch_loss, epoch_obj, epoch_cons = 0, 0, 0

        for batch in tqdm(data_loader, desc="Train"):
            batch = batch.to(self.device)

            output, _  = self.model.forward(batch)
            output = output.reshape(-1, 1)
            logits = unbatch(output, batch=batch.variable_features_batch)

            batch_loss, info = self.train_loss_computer.compute(logits, batch, is_training=True)
            
            epoch_loss += batch_loss.item() 
            epoch_obj += info['batch_obj'].item()
            epoch_cons += info['batch_cons'].item() 
            num_samples += len(batch)

            batch_loss = batch_loss / len(batch)

            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.lr_scheduler.step()
        self.train_loss_computer.update_mu(epoch_cons, num_samples)
        self.val_loss_computer.mu = self.train_loss_computer.mu

        return epoch_loss / num_samples

    @torch.no_grad()
    def validate_epoch(self, data_loader):
        self.model.eval()
        num_samples = 0
        epoch_loss, epoch_obj, epoch_cons, epoch_best = 0, 0, 0, 0

        for batch in tqdm(data_loader, desc="Valid"):
            batch = batch.to(self.device)

            output, cons_o = self.model(batch)
            output = output.reshape(-1, 1)

            logits = unbatch(output, batch=batch.variable_features_batch)

            batch_loss, info = self.val_loss_computer.compute(logits, batch, is_training=False)

            batch_size = len(batch)
            num_samples += batch_size

            epoch_loss += batch_loss.item() 
            epoch_obj += info['batch_obj'].item()
            epoch_cons += info['batch_cons'].item() 
            epoch_best += info['batch_best']

        return epoch_best / num_samples