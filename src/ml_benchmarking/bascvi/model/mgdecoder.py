import collections
from typing import Optional
import torch
import torch.nn as nn

class MacroGeneDecoder(nn.Module):
    """
    Decoder for the macrogene benchmark model
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.Softmax(dim=-1),
        )
       
        # dropout
        self.px_dropout_decoder = nn.Linear(n_input, n_output)

        # dispersion parameter
        self.px_r = torch.nn.Parameter(torch.randn(n_output))

    def forward(
        self,
        z: torch.Tensor,
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
        
        px_scale = self.px_scale_decoder(z)

        px_dropout = self.px_dropout_decoder(z)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        clamp_max = torch.exp(torch.tensor([12.0])).item()
        if library is None:
            px_rate = px_scale.clamp(max=clamp_max)  # torch.clamp( , max=12)
        else:
            px_rate = (torch.exp(library) * px_scale).clamp(max=clamp_max)  # torch.clamp( , max=12)

        px_r = torch.exp(self.px_r)
        return px_scale, px_rate, px_dropout, px_r


