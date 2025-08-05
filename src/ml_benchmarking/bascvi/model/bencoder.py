import collections
import torch
import torch.nn as nn
from torch.distributions import Normal

def reparameterize_gaussian(mu, var):
    if torch.isnan(mu).any() or torch.isnan(var).any():
        print("NaN detected in reparameterization: mu or var contains NaN!")
        raise ValueError("NaN detected in reparameterization!")

    var = torch.clamp(var, min=1e-6)  # Ensure variance is positive
    return Normal(mu, var.sqrt()).rsample()


class BEncoder(nn.Module):
    """
    Encodes high-dimensional gene expression data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_layers`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
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
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
    ):
        super().__init__()

        self.var_eps = var_eps
        layers_dim = [n_input] + (n_layers) * [n_hidden]
        
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in,
                                n_out,
                            ),
                            nn.BatchNorm1d(n_out, eps=1e-3), #nn.LayerNorm(n_out),
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
            nn.BatchNorm1d(n_hidden, eps=1e-3), # nn.LayerNorm(n_hidden),
            nn.ReLU(),
            )
            
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """

        for layer in self.encoder:
            # Forward through layer
            x = layer(x)

        # Final block encoder
        x = self.b_encoder(x)

        x_pred = x
        
        # Parameters for latent distribution
        q = x
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + self.var_eps
        
        # Sample from the distribution
        latent = reparameterize_gaussian(q_m, q_v)
        
        return q_m, q_v, latent, x_pred
