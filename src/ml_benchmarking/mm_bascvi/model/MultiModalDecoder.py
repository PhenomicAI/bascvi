from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_benchmarking.mm_bascvi.model.distributions import ZeroInflatedNegativeBinomial
class SharedDecoder(nn.Module):
    """
    A single decoder that reconstructs gene-expression data (or any vector data)
    from a latent embedding, regardless of modality.

    Condition on modality by:
      1. Learning an embedding for each modality, 
      2. Concatenating it to the latent vector 'z', 
      3. Passing that concatenated vector through the MLP.

    This ensures the decoder can adapt to each modality’s nuances within one network,
    but uses one set of "core" parameters overall.
    """
    def __init__(
        self,
        latent_dim: int,
        n_input: int,
        batch_level_sizes: list,
        hidden_dims: list,
        batch_embedding_dim: int = 16,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_input = n_input

        # Learn an embedding vector for each batch level
        self.batch_level_embeddings = nn.ModuleList([
            nn.Embedding(batch_level_sizes[i], batch_embedding_dim)
            for i in range(len(batch_level_sizes))
        ])

        # A simple MLP that maps from latent -> reconstructed data
        dims = [latent_dim + batch_embedding_dim * len(batch_level_sizes)] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_input),
            nn.Softmax(dim=-1),
        )
       
        # dropout
        self.px_dropout_decoder = nn.Linear(hidden_dims[-1], n_input)

        self.px_r = torch.nn.Parameter(torch.randn(n_input))


    def forward(self, z: torch.Tensor, batch_idx: torch.Tensor, library: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          z: [batch_size, latent_dim]  The shared latent embedding
          batch_idx: [batch_size, n_batch_levels]   Integer labels for each batch level
          library: [batch_size]        Optional library size
        Returns:
           3-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        batch_emb_list = [self.batch_level_embeddings[i](batch_idx[:, i]) for i in range(batch_idx.shape[1])]

        # Concatenate along the feature dimension
        z = torch.cat([z] + batch_emb_list, dim=1)

        # Pass through the MLP to get reconstruction
        px = self.decoder(z)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)

        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        clamp_max = torch.exp(torch.tensor([12.0])).item()
        
        if library is None:
            px_rate = px_scale.clamp(max=clamp_max, min=0)  # torch.clamp( , max=12)
        else:
            px_rate = (torch.exp(library) * px_scale).clamp(max=clamp_max, min=0)  # torch.clamp( , max=12)

        # Reconstructed expression using ZINB parameters
        x_reconstructed = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=torch.exp(self.px_r), zi_logits=px_dropout).sample()

        return x_reconstructed, (px_rate, torch.exp(self.px_r),px_dropout)
