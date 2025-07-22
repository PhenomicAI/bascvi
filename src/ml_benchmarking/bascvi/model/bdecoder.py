import collections
from typing import Optional
import torch
import torch.nn as nn

class BDecoder(nn.Module):
    """
    Decodes gene expression from latent space of ``n_input`` dimensions to ``n_output``dimensions.
    Decoder learns values for the parameters of the ZINB distribution over each gene
    Uses a fully-connected neural network of ``n_layers`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_batch
        Number of batches, either no. of batches in input data or batch embedding dimension
    n_output
        The dimensionality of the output (gene space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.n_batch = n_batch
        
        
        self.b_decoder = nn.Sequential(
            nn.Linear(
                    n_input,
                    n_hidden,
                    ),
            nn.LayerNorm(n_hidden, eps=1e-5), # nn.LayerNorm(n_hidden),
            nn.ReLU(),
            )
        
        
        layers_dim = [n_hidden] + (n_layers) * [n_hidden]
        self.px_decoder = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + n_batch,
                                n_out,
                            ),
                            nn.LayerNorm(n_out, eps=1e-5), # nn.LayerNorm(n_out),
                            nn.ReLU(),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )
       
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        batch_emb: torch.Tensor,
        library: Optional[torch.Tensor] = None,
    ):
        """
        The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        batch_emb
            batch_emb
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        
        z = self.b_decoder(z)
        z_pred = z
        
        for layer in self.px_decoder:
            z = torch.cat((z, batch_emb), dim=-1)
            z = layer(z)
        px = z
        px_scale = self.px_scale_decoder(px)

        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        clamp_max = torch.exp(torch.tensor([11.0])).item()
        if library is None:
            px_rate = px_scale.clamp(max=clamp_max)  # torch.clamp( , max=12)
        else:
            px_rate = (torch.exp(library) * px_scale).clamp(max=clamp_max)  # torch.clamp( , max=12)
        return px_scale, px_rate, px_dropout, z_pred


