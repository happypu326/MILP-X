import torch
import torch.nn.functional as F

def gumbel_sample(logits, N=15, tau=1.0):
    logits = logits.reshape(-1, 1)
    logits = logits.repeat(N, 1, 1)
    logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)[:,:,1]

class DiffILOLossComputer:
    def __init__(self, config):
        self.mu = config.get('mu_init', 5.0)
        self.mu_step_size = config.get('mu_step_size', 0.01)
        self.mu_value = config.get('mu_value', 1.0)
        self.mu_max = config.get('mu_max', 5.0)
        self.mu_min = config.get('mu_min', 0.1)
        self.loss_config = config.get('loss_config', 'normalize')
        self.num_samples = config.get('num_samples', 15)

    def compute(self, model_output, batch, is_training=False):
        batch = batch.to_data_list()
        device = model_output[0].device
        
        batch_loss = torch.zeros(1, device=device)
        batch_obj = torch.zeros(1, device=device) 
        batch_cons = torch.zeros(1, device=device)
        batch_best = 0
        batch_best_obj = 0
        batch_mean_obj = 0

        # Process each graph in batch
        if is_training:
            for i, g in enumerate(batch):
                # Sample solutions
                x = gumbel_sample(model_output[i], self.num_samples, 1.0).float().reshape(self.num_samples, -1)
                
                # Get problem matrices
                A = g.A.cuda()
                b = g.b.cuda()
                c = g.c.cuda()

                # Calculate objectives
                p = torch.sigmoid(model_output[i])
                obj = (p * c).sum()
                cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)
                entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum()
                
                # Calculate loss based on config
                if self.loss_config == "normalize" and torch.norm(c) > 0:
                    num_nonzero = torch.count_nonzero(cons_pos)
                    loss = obj / torch.norm(c) + self.mu * (1 / torch.norm(A, dim=1) * cons_pos.squeeze()).sum() / num_nonzero if num_nonzero > 0 else obj / torch.norm(c)
                elif self.loss_config == "sum":
                    loss = obj + self.mu * cons_pos.sum()
                elif self.loss_config == "mean":
                    loss = obj + self.mu * cons_pos.mean()
                elif self.loss_config == "nonzero_mean":
                    num_nonzero = torch.count_nonzero(cons_pos)
                    loss = obj + self.mu * cons_pos.mean() / num_nonzero if num_nonzero > 0 else obj

                with torch.no_grad():
                    xx = gumbel_sample(model_output[i], 500, 1.0).float().reshape(500, -1)
                    idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
                    best = (xx @ c)[idx].min().item() if len(idx) > 0 else float('inf')
                    best_obj = (xx @ c).min().item()
                    mean_obj = (xx @ c).mean().item()

                batch_obj += obj
                batch_cons += cons_pos.sum()
                batch_loss += loss
                batch_best += best
                batch_best_obj += best_obj
                batch_mean_obj += mean_obj
                
                batch_info = {
                    'batch_obj':batch_obj,
                    'batch_cons':batch_cons,
                    'batch_best':batch_best,
                    'batch_best_obj':batch_best_obj,
                    'batch_mean_obj':batch_mean_obj
                }
        else:
            # Calculate loss
            for g, logit in zip(batch, model_output):
                # Sample solutions
                x = gumbel_sample(logit, self.num_samples, 1.0).float().reshape(self.num_samples, -1)
                
                # Get problem matrices/vectors
                A, b, c = [tensor.cuda() for tensor in (g.A, g.b, g.c)]

                # Calculate metrics
                p = torch.sigmoid(logit)
                obj = (p * c).sum()
                cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)
                
                # Calculate loss
                loss = obj + self.mu * cons_pos.sum()

                # Additional sampling for statistics
                xx = gumbel_sample(logit, 500, 1.0).float().reshape(500, -1)
                feasible_idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
                best = (xx @ c)[feasible_idx].min().item() if len(feasible_idx) > 0 else 1e3
            
                batch_obj += obj
                batch_cons += cons_pos.sum()
                batch_loss += loss
                batch_best += best
                batch_best_obj += (xx @ c).min().item()
                batch_mean_obj += (xx @ c).mean().item()

                # Update batch metrics
                batch_info = {
                    'batch_obj':batch_obj,
                    'batch_cons':batch_cons,
                    'batch_best':batch_best,
                    'batch_best_obj':batch_best_obj,
                    'batch_mean_obj':batch_mean_obj
                }

        return batch_loss, batch_info

    def update_mu(self, epoch_cons, num_samples):
        self.mu = self.mu + self.mu_step_size * (epoch_cons / num_samples - self.mu_value)
        self.mu = max(min(self.mu, self.mu_max), self.mu_min)
