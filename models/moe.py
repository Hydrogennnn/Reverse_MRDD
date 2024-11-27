import torch
import torch.nn as nn
import torch.nn.functional as F


class Moe(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim):
        super(Moe, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = (input_dim+output_dim) * 2 //3

        self.experts = nn.ModuleList([
                nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            ) for _ in range(self.num_experts)
        ])
        # slots = output_dim
        self.phi = nn.Parameter(torch.randn(input_dim, num_experts, self.output_dim))

    def soft_moe_layer(self, x):
        logits = torch.einsum('md,dnp->mnp', x, self.phi)
        # print('log:', logits.shape)
        D = F.softmax(logits, dim=1)  # Dispatch weights
        C = F.softmax(logits.view(logits.shape[0], -1), dim=-1)  # Combine weights
        C = C.view_as(logits)
        Xs = torch.einsum('md,mnp->npd', x, D)  # Weighted input slots
        # print('Xs:', Xs.shape)
        Ys = torch.stack([f_i(Xs[i, :, :]) for i, f_i in enumerate(self.experts)], dim=0)  # Expert outputs
        # print('C', C.shape)
        Y = torch.einsum('npd,mnp->md', Ys, C)  # Combine expert outputs
        # print('Y', Y.shape)
        return Y





    def forward(self, x):
        x = self.soft_moe_layer(x) # (batch_size, num_experts, output_dim)
        x = torch.flatten(x,start_dim=1)
        return x




