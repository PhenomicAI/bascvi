import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    """
    This encoder produces latent embedding (VAE style).
    """
    def __init__(self, n_genes: int, latent_dim: int, hidden_dims: list = [128, 64]):
        super().__init__()
        
        layers = []
        dims = [n_genes] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Return the latent plus (mu, logvar) for VAE losses
        return z, (mu, logvar)
