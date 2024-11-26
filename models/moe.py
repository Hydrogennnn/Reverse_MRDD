import torch
import torch.nn as nn
import torch.nn.functional as F


class Moe(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim):
        super(Moe, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = (input_dim+input_dim) * 2 //3

        self.experts = nn.ModuleList([
                nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.input_dim)
            ) for _ in range(self.num_experts)
        ])

        self.output_layer = nn.Linear(self.num_experts * self.input_dim, self.output_dim)
        self.phi = nn.Parameter(torch.randn(input_dim, num_experts, self.hidden_dim))

    def soft_moe_layer(self, x):
        logits = torch.einsum('bmd,dnp->bmnp', x, self.phi)
        # print('log:', logits.shape)
        D = F.softmax(logits, dim=1)  # Dispatch weights
        C = F.softmax(F.softmax(logits, dim=2), dim=3)  # Combine weights
        Xs = torch.einsum('bmd,bmnp->bnpd', x, D)  # Weighted input slots
        # print('Xs:', Xs.shape)
        Ys = torch.stack([f_i(Xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1)  # Expert outputs
        # print('Ys', Ys.shape)
        # print('C', C.shape)
        Y = torch.einsum('bnpd,bmnp->bmd', Ys, C)  # Combine expert outputs
        # print('Y', Y.shape)
        return Y





    def forward(self, x):
        x = self.soft_moe_layer(x)
        x = torch.flatten(x,start_dim=1)
        x = self.output_layer(x)
        # print('out x', x.shape)
        return x




