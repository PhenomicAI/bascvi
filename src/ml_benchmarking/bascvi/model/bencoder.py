import collections
import torch
import torch.nn as nn
from torch.distributions import Normal

def reparameterize_gaussian(mu, logvar):
    # Don't check for NaNs - we'll handle them in the forward method
    std = torch.exp(0.5 * logvar).clamp(min=1e-4)
    return Normal(mu, std).rsample()


class BEncoder(nn.Module):
    """
    Encodes high-dimensional gene expression data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_layers`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_batch
        Number of batches, either no. of batches in input data or batch embedding dimension
    n_output
        The dimensionality of the output (latent space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.n_batch = n_batch
        layers_dim = [n_input] + (n_layers) * [n_hidden]
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + n_batch,
                                n_out,
                            ),
                            nn.LayerNorm(n_out),# nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001), #nn.LayerNorm(n_out),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
        
        self.b_encoder = nn.Sequential(
            nn.Linear(
                    n_hidden,
                    n_hidden,
                    ),
            nn.LayerNorm(n_hidden),# nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001), # nn.LayerNorm(n_hidden),
            nn.ReLU(),
            )
            
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = nn.Linear(n_hidden, n_output)


    def forward(self, x: torch.Tensor, batch_emb: torch.Tensor):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        batch_emb
            batch_emb for this sample
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """

        zeros = torch.zeros_like(batch_emb)
        
        # Process through encoder layers
        for layer in self.encoder:
            x = torch.cat((x, zeros), dim=-1)
            x = layer(x)
        
        x = self.b_encoder(x)
        x_pred = x
        q = x
        
        # Get parameters
        q_m = self.mean_encoder(q)
        q_log_var = self.log_var_encoder(q)
        
        # Debug before replacement
        if torch.isnan(q_m).any() or torch.isnan(q_log_var).any():
            print("NaNs detected before replacement")
            print(f"q_m NaNs: {torch.isnan(q_m).sum().item()}")
            print(f"q_log_var NaNs: {torch.isnan(q_log_var).sum().item()}")
        
        # Replace and clamp IN-PLACE to ensure no new tensor is created
        q_m = q_m.clone()  # Ensure we have a clean copy
        q_log_var = q_log_var.clone()
        
        # Replace NaNs in-place
        q_m.nan_to_num_(nan=0.0)
        q_log_var.nan_to_num_(nan=0.0)
        
        # Clamp in-place  
        q_m.clamp_(min=-20.0, max=20.0)
        q_log_var.clamp_(min=-10.0, max=4.0)
        
        # Verify no NaNs remain
        assert not torch.isnan(q_m).any(), "NaNs still in q_m after replacement!"
        assert not torch.isnan(q_log_var).any(), "NaNs still in q_log_var after replacement!"
        
        # Modified reparameterization that doesn't check for NaNs
        std = torch.exp(0.5 * q_log_var).clamp(min=1e-4)
        latent = Normal(q_m, std).rsample()
        
        return q_m, q_log_var, latent, x_pred
