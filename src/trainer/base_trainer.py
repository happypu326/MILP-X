import os
import time
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader,
        val_dataloader,
        train_loss_computer,
        val_loss_computer,
        config: dict
    ):
        """
        Args:
            model: The model to be trained
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            train_loss_computer: Training loss calculator (can be None)
            val_loss_computer: Validation loss calculator (can be None)
            config: Configuration dictionary containing all training parameters
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_loss_computer = train_loss_computer
        self.val_loss_computer = val_loss_computer
        self.config = config
        
        self.device = torch.device(config.get('device', 'cpu'))
        self.logger = config.get('logger', None)
        
        self.num_epochs = config.get('num_epochs', 100)
        self.save_top_k = config.get('save_top_k', 3)
        self.patience = config.get('patience', 100)
        self.min_delta = config.get('min_delta', 1e-5)
        
        save_dir = config.get('model_save_dir', './pretrain_models')
        log_dir = config.get('log_save_dir', './train_logs')
        self.method_type = config.get('method_type', 'PS')
        self.problem_type = config.get('problem_type', '')
        
        self.save_dir = os.path.join(save_dir, self.method_type, self.problem_type)
        self.log_dir = os.path.join(log_dir, self.method_type, self.problem_type)
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model.to(self.device)
        
        if self.method_type == "DiffILO":
            self.lr_i = config.get('lr_i', 0.0001)
            self.lr_o = config.get('lr_o', 0.0001)
            self.optimizer = config.get('optimizer','adam')
            self.weight_decay = config.get('weight_decay', 0.0)
            self.momentum = config.get('momentum', 0.9)
            
            output_params = {id(p) for layer in (self.model.vars_output_layer, self.model.cons_output_layer) 
                        for p in layer.parameters()}
            other_params = [p for p in self.model.parameters() if id(p) not in output_params]
            
            params_dict = [
                {'params': self.model.vars_output_layer.parameters(), 'lr': self.lr_o},
                {'params': other_params, 'lr': self.lr_i}
            ]
            
            optimizer_map = {
                "adam": lambda: torch.optim.Adam(params_dict, weight_decay=self.weight_decay),
                "sgd": lambda: torch.optim.SGD(params_dict, momentum=0.9, weight_decay=1e-4)
            }
            self.optimizer = optimizer_map[self.optimizer]()
        else:
            self.lr = config.get('lr', 1e-5)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        scheduler_map = {
            "exp": lambda: torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=config.get("gamma", 0.9)
            ),
            "cos": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("cos_T", 200),
                eta_min=config.get("cos_min", 0.0)
            ),
            "cosrestart": lambda: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get("cos_T", 200),
                T_mult=config.get("cos_T_mult", 1), 
                eta_min=config.get("cos_min", 0.0)
            )
        }

        scheduler_type = config.get("lr_scheduler", None)
        if scheduler_type != None:
            self.lr_scheduler = scheduler_map[scheduler_type]()
        else:
            self.lr_scheduler = None

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        self.patience_counter = 0
    
    @abstractmethod
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Subclass implementation should directly use self.train_dataloader
            
        Returns:
            Average training loss
        """
        pass
    
    @abstractmethod
    def validate_epoch(self) -> float:
        """
        Validate for one epoch
    
        Subclass implementation should directly use self.val_dataloader
            
        Returns:
            Average validation loss
        """
        pass
    
    def train(self):
        best_models = []
        save_name = self.config.get('save_name', 'model')
        
        log_file = open(os.path.join(self.log_dir, f'{save_name}_train.log'), 'wb')
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                start_time = time.time()
                
                train_loss = self.train_epoch(self.train_dataloader)
                self.train_history.append(train_loss)
                
                val_loss = self.validate_epoch(self.val_dataloader)
                self.val_history.append(val_loss)
                
                epoch_time = time.time() - start_time
                
                self._log_epoch(epoch, train_loss, val_loss, epoch_time, log_file)
                
                should_save_topk = (
                    len(best_models) < self.save_top_k
                    or val_loss < best_models[-1][0]
                )
                if should_save_topk:
                    model_path = os.path.join(
                        self.save_dir,
                        f'{save_name}_epoch{epoch}_val{val_loss:.6f}.pth'
                    )
                    torch.save(self.model.state_dict(), model_path)
                    best_models.append((val_loss, epoch, model_path))
                    best_models.sort(key=lambda x: x[0])

                    if len(best_models) > self.save_top_k:
                        _, _, worst_path = best_models.pop()
                        try:
                            os.remove(worst_path)
                        except FileNotFoundError:
                            pass
                
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{save_name}_model_best.pth'))
                    log_entry_best = f'@epoch{epoch}   New best model saved with validation loss: {val_loss:.6f}\n'
                    log_file.write(log_entry_best.encode())
                else:
                    self.patience_counter += 1
                    log_entry_patience = f'@epoch{epoch}   No improvement. Patience counter: {self.patience_counter}/{self.patience}\n'
                    log_file.write(log_entry_patience.encode())
                
                last_path = os.path.join(self.save_dir, f'{save_name}_model_last.pth')
                torch.save(self.model.state_dict(), last_path)

                if self.patience_counter >= self.patience:
                    early_stop_msg = f'\nEarly stopping triggered at epoch {epoch}. No improvement for {self.patience} consecutive epochs.\n'
                    log_file.write(early_stop_msg.encode())
                    log_file.flush()
                    self._log_message(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss:.6f}")
                    break
        
        finally:
            log_file.close()        

        if self.patience_counter < self.patience:
            self._log_message('Training completed!')
        else:
            self._log_message(f"Training stopped early at epoch {self.current_epoch} due to no improvement.")
        self._log_message(f'Best validation loss: {self.best_val_loss:.6f}')
    
    def _log_epoch(self, epoch, train_loss, val_loss, epoch_time, log_file):
        if self.logger:
            self.logger.info(
                f"Epoch {epoch} - Train loss: {train_loss:.6f}, "
                f"Valid loss: {val_loss:.6f} (Time: {epoch_time:.2f}s)"
            )
            self.logger.log_scalar('epoch/train_loss', train_loss, step=epoch)
            self.logger.log_scalar('epoch/val_loss', val_loss, step=epoch)
        else:
            print(f"@epoch{epoch}   Train loss:{train_loss:.6f}   Valid loss:{val_loss:.6f}    TIME:{epoch_time:.2f}")

        log_entry = f'@epoch{epoch}   Train loss:{train_loss}   Valid loss:{val_loss}    TIME:{epoch_time}\n'
        log_file.write(log_entry.encode())
        log_file.flush()

    def _log_message(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
