import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


# SIGNAL_NOISE = 0.2
# SIGNAL_POWER = 3

# data_dim = 3
# num_instances = 20000


def sample_data(mu, var):
    eps = torch.rand_like(mu)
    return mu + eps*var

# def gen_x(num, dim):
#     return np.random.normal(0., np.sqrt(SIGNAL_POWER), [num, dim])


# def gen_y(x, num, dim):
#     return x + np.random.normal(0., np.sqrt(SIGNAL_NOISE), [num, dim])


# def true_mi(power, noise, dim):
#     return dim * 0.5 * np.log2(1 + power/noise)


# mi = true_mi(SIGNAL_POWER, SIGNAL_NOISE, data_dim)
# print('True MI:', mi)
n_epoch = 500


class MINE(nn.Module):
    def __init__(self,  x_dim, y_dim, device, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(x_dim+y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.device = device
        
    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        inputs = inputs.to(self.device)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss

class Estimator():
    def __init__(self, x_dim, y_dim, device):
        self.net = MINE(x_dim, y_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.device = device
    
    def learning_loss(self, mu_x, var_x, mu_y, var_y):
        loss = 0.
        for epoch in tqdm(range(n_epoch)):
            x_sample = sample_data(mu_x, var_x)
            y_sample = sample_data(mu_y, var_y)
            x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)
            loss = self.net(x_sample, y_sample)
            self.optimizer.zero_grad()
            
            if epoch ==n_epoch-1:
                loss.backward(retain_graph=true)
            else:
                loss.backward()
            self.optimizer.step()
        
        return -loss
    
        
    