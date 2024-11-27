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

    def soft_moe_layer(self, X):
        # X: [batch_size, views, input_dim]
        m = X.shape[1]
        batch_size = X.shape[0]
        # Compute logits for dispatch and combine weights
        logits = torch.einsum('bmd,dnp->bmnp', X, self.phi)  # [batch_size, m, num_experts, p]

        # Compute dispatch weights (per sample)
        D = F.softmax(logits, dim=1)  # [batch_size, m, num_experts, p]

        # Compute combine weights (shared across samples)
        C = F.softmax(logits.view(batch_size, m, -1), dim=-1)
        C = torch.mean(C, dim=1, keepdim=True)  # [batch_size, 1, n*p]
        C = C.view(batch_size, 1, self.num_experts, self.hidden_dim) #[batch_size, 1, n, p]

        # Dispatch inputs to experts
        Xs = torch.einsum('bmd,bmnp->bnpd', X, D)  # [batch_size, num_experts, p, input_dim]
        # Apply each expert function
        Ys = torch.stack([self.experts[i](Xs[:,i ,:, :]) for i in range(self.num_experts)],
                         dim=1)  # [batch_size, num_experts, p, output_dim]
        # Combine expert outputs (shared weights across all samples)
        Y = torch.einsum('bnpd,bmnp->bmd', Ys, C.repeat(1, m, 1, 1))  # [batch_size, m, output_dim]
        # Fusion: Aggregate across all samples
        Y_fused = Y.mean(dim=1)  # [batch_size, output_dim]
        return Y_fused

    def forward(self, x):
        x = self.soft_moe_layer(x)
        # print('out x', x.shape)

        return x




