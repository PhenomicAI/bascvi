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
        n_genes: int,
        num_modalities: int,
        hidden_dims: list = [128, 64],
        modality_embedding_dim: int = 16,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_genes = n_genes

        # Learn an embedding vector for each modality
        self.modality_embedding = nn.Embedding(num_modalities, modality_embedding_dim)

        # A simple MLP that maps from latent -> reconstructed data
        dims = [latent_dim + modality_embedding_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_genes),
            nn.Softmax(dim=-1),
        )
       
        # dropout
        self.px_dropout_decoder = nn.Linear(hidden_dims[-1], n_genes)

        self.px_r = torch.nn.Parameter(torch.randn(n_genes))


    def forward(self, z: torch.Tensor, modality_vec: torch.Tensor, library: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          z: [batch_size, latent_dim]  The shared latent embedding
          modality_vec: [batch_size]   Integer labels for each modality
          library: [batch_size]        Optional library size
        Returns:
           3-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        modality_emb = self.modality_embedding(modality_vec)
        # Concatenate along the feature dimension
        z = torch.cat([z, modality_emb], dim=1)

        # Pass through the MLP to get reconstruction
        px = self.decoder(z)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)

        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        clamp_max = torch.exp(torch.tensor([12.0])).item()
        
        if library is None:
            px_rate = px_scale.clamp(max=clamp_max)  # torch.clamp( , max=12)
        else:
            px_rate = (torch.exp(library) * px_scale).clamp(max=clamp_max)  # torch.clamp( , max=12)

        # Reconstructed expression using ZINB parameters
        x_reconstructed = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=torch.exp(self.px_r), zi_logits=px_dropout).sample()

        return x_reconstructed, (px_rate, torch.exp(self.px_r),px_dropout)
