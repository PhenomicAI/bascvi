import torch
import torch.nn as nn


class GeneNorm(nn.Module):
    def __init__(self, num_genes):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(1, num_genes))
        self.shift = nn.Parameter(torch.zeros(1, num_genes))
        self.log_scaling_factor = torch.log(torch.tensor(10000.0)) # nn.Parameter(torch.log(torch.tensor(10.0)))


    def forward(self, x):
        # x is a tensor of shape (batch_size, num_genes)
        x_sum = x.sum(dim=1, keepdim=True)
        x_norm = x / (x_sum + 1e-6)

        # Normalize the input
        x = torch.log1p(torch.exp(self.log_scaling_factor) * x_norm)

        x = x * self.scale + self.shift

        x = torch.clamp(x, min=-10.0, max=10.0)

        return x