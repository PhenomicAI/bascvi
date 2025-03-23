import torch
import torch.nn as nn
from torch.distributions import Normal


class VAEEncoder(nn.Module):
    """
    This encoder produces latent embedding (VAE style).
    """
    def __init__(self, n_input: int, latent_dim: int, hidden_dims: list = [128, 64]):
        super().__init__()
        
        layers = []
        dims = [n_input] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu, logvar):
        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("NaN detected in reparameterization: mu or logvar contains NaN!")
            raise ValueError("NaN detected in reparameterization!")
        
        std = torch.exp(0.5 * logvar)  # Convert logvar to standard deviation
        return Normal(mu, std).rsample()

    def forward(self, x: torch.Tensor):

        # Normalize the input
        x = torch.log(1 + 10000 * x / (x.sum(dim=1, keepdim=True) + 1e-6))

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Return the latent plus (mu, logvar) for VAE losses
        return z, (mu, logvar)
