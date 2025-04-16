import torch
import torch.nn as nn

from .VAEEncoder import VAEEncoder
from .BatchDiscriminator import BatchDiscriminator
from .CrossAttention import ModalityCrossAttention
from .CrossAttention import LearnedTempCellTypeCrossAttention
from .MultiModalDecoder import SharedDecoder
from .Norm import GeneNorm

import itertools
from typing import List

import random

# -----------------------
# Main MMBAscVI model 
# -----------------------
class MMBAscVI(nn.Module):
    """
    High-level model orchestrating:
      1. Multiple ModalityExperts (scrna_seq, spatial rna, bulk rna) 
      2. Multiple CellTypeExperts
      3. 2 ModalityPredictors (for classification of z from modality experts and celltype experts)
      4. Cross-attention blocks
      5. One shared decoder for final reconstruction
    """
    def __init__(
        self,
        n_input: int,
        batch_level_sizes: List[int],
        latent_dim: int = 16,
        ct_latent_dim: int = 16,
        n_cell_type_experts: int = 10,
        modality_encoder_hidden_dims: list = [128, 128],
        ct_encoder_hidden_dims: list = [128, 128],
        predictor_hidden_dims: list = [128, 128, 128],
        decoder_hidden_dims: list = [128, 128, 128],
        residual_weight: float = 1.0,
        bulk_id: int = None,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.batch_level_sizes = batch_level_sizes

        self.bulk_id = bulk_id

        self.n_batch = sum(batch_level_sizes)

        self.n_modalities = batch_level_sizes[0]
        self.n_studies = batch_level_sizes[1]
        self.n_samples = batch_level_sizes[2]

        self.latent_dim = latent_dim
        self.ct_latent_dim = ct_latent_dim
        self.n_cell_type_experts = n_cell_type_experts


        # 0) GeneNorm per modality
        self.gene_norm_list = nn.ModuleList([
            GeneNorm(n_input)
            for _ in range(self.n_modalities)
        ])

        # 1) Modality Experts
        self.modality_experts = nn.ModuleList([
            VAEEncoder(n_input, latent_dim, modality_encoder_hidden_dims)
            for _ in range(self.n_modalities)
        ])

        # 2) Batch Discriminators
        self.batch_discriminator_celltype_list = nn.ModuleList([
            BatchDiscriminator(ct_latent_dim, d, predictor_hidden_dims)
            for d in self.batch_level_sizes
        ])

        self.batch_discriminator_final_list = nn.ModuleList([
            BatchDiscriminator(latent_dim, d, predictor_hidden_dims)
            for d in self.batch_level_sizes
        ])

        # 3) Cell Type Experts
        self.celltype_experts = nn.ModuleList()
        for i in range(n_cell_type_experts):
            # Set a different seed for each expert
            torch.manual_seed(42 + i)  # Use different seeds
            expert = VAEEncoder(n_input, ct_latent_dim, ct_encoder_hidden_dims, subset_genes_prop=random.uniform(0.6, 0.8))
            self.celltype_experts.append(expert)

        # Reset the random seed after initialization
        torch.manual_seed(torch.initial_seed())

        # 4) Cross-Attention 
        self.celltype_cross_attention = LearnedTempCellTypeCrossAttention(
                    latent_dim=latent_dim,
                    ct_latent_dim=ct_latent_dim,
                    num_modalities=self.n_modalities,
                    residual_weight=residual_weight
                )
        
        # 5) Shared decoder
        self.shared_decoder = SharedDecoder(
            latent_dim=latent_dim,
            n_input=n_input,
            batch_level_sizes=self.batch_level_sizes,
            hidden_dims=decoder_hidden_dims        
            )
        
        self.g_params = itertools.chain(
            self.modality_experts.parameters(),
            self.celltype_experts.parameters(),
            self.celltype_cross_attention.parameters(),
            self.shared_decoder.parameters(),
            )
        
        self.d_params = itertools.chain(
            self.batch_discriminator_celltype_list.parameters(),
            self.batch_discriminator_final_list.parameters()
            )


    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor, bulk_mask: torch.Tensor):
        """
        Args:
            x:           [batch_size, n_input]
            batch_idx:  [batch_size, n_batch_levels]
        Returns:
            A dictionary of intermediate outputs, including:
                Embeddings:
                    - z_modality_list, z_modality_refined
                    - z_celltype_list, z_cell_refined
                Reconstruction:
                    - recon_x
                Attention weights:
                    - cell_attn_weights
                Logits:
                    - modality_logits_list
                    - refined_modality_logits
                    - celltype_logits_list
                VAE parameters:
                    - celltype_vae_params
                    - modality_vae_params
        """

        modality_idx = batch_idx[:, 0]
        study_idx = batch_idx[:, 1]
        sample_idx = batch_idx[:, 2]

        if torch.isnan(x).any():
            raise ValueError("NaN detected in input batch!")

        # ----- 0) GeneNorm per modality -----
        x_norm_list = []
        for i in range(self.n_modalities):
            x_norm_list.append(self.gene_norm_list[i](x))

        x_norm_true = torch.stack(x_norm_list, dim=1)
        x_norm_true = x_norm_true[torch.arange(x.size(0)), modality_idx]

        if torch.isnan(x_norm_true).any():
            raise ValueError("NaN detected in x_norm_true!")

        # ----- 1) Run each ModalityExpert (Encoders) -----
        # using modality_idx run the modality experts, only run the modality experts for the given modality_idx

        z_modality_list = []
        modality_vae_params = []
        # modality_logits_list = []

        for i in range(self.n_modalities):
            z_m, (mu_m, logvar_m) = self.modality_experts[i](x_norm_list[i])
            z_modality_list.append(z_m)
            modality_vae_params.append(torch.stack([mu_m, logvar_m], dim=0))

        # Convert z_modality_list to tensor
        z_modality_list = torch.stack(z_modality_list, dim=1) # -> [batch_size, num_modalities, latent_dim]
        modality_vae_params = torch.stack(modality_vae_params, dim=1) # -> [2, num_modalities, batch_size, latent_dim]
        modality_vae_params = modality_vae_params.permute(2, 1, 0, 3) # -> [batch_size, num_modalities, 2, latent_dim]


        # Gather the correct entries for each batch using idx
        z_modality_refined = z_modality_list[torch.arange(x.size(0)), modality_idx]  # shape: [256, 10]
        modality_vae_params = modality_vae_params[torch.arange(x.size(0)), modality_idx] # shape: [256, 2, latent_dim]

        # ----- 3) Cell Type Experts (Encoders) -----
        z_celltype_list = []
        celltype_vae_params = []
        disc_ct_logits_list = []

        for expert in self.celltype_experts:
            z_c, (mu_c, logvar_c) = expert(x_norm_true)
            z_celltype_list.append(z_c)
            celltype_vae_params.append((mu_c, logvar_c))

            # Predict each batch level from celltype expert
            disc_ct_logits_list.append([self.batch_discriminator_celltype_list[i](z_c) for i in range(len(self.batch_level_sizes))])

        z_celltype_stack = torch.stack(z_celltype_list, dim=1)
        # => [batch_size, num_cell_type_experts, latent_dim]

        # ----- 4) Cell-Type Cross-Attention -----
        z_cell_refined, cell_attn_weights = self.celltype_cross_attention(
            z_modality_refined,
            z_celltype_stack,
            modality_idx # needed for per-modality temperature
        )
        # => [batch_size, latent_dim]

        # (b) Also predict batch levels from the final embedding
        disc_z_logits_list = [
            self.batch_discriminator_final_list[i](z_cell_refined)
            for i in range(len(self.batch_level_sizes)) 
        ]

        # ----- 5) Shared Decoder -----
        # Here we decode from the final 'z_cell_refined'
        x_reconstructed, x_decoder_zinb_params = self.shared_decoder(z_cell_refined, batch_idx, bulk_mask=bulk_mask)

        return {
            # Modality expert:
            "modality_vae_params":   modality_vae_params,
            "z_modality_refined":    z_modality_refined,

            # Cell type experts:
            "z_celltype_list":       z_celltype_list,
            "celltype_vae_params":   celltype_vae_params,
            "z_cell_refined":        z_cell_refined,
            "cell_attn_weights":     cell_attn_weights,

            # Batch discriminators:
            "disc_ct_logits_list":  disc_ct_logits_list,
            "disc_z_logits_list":   disc_z_logits_list,

            # Shared decoder parameters:
            "x_reconstructed": x_reconstructed,
            "x_decoder_zinb_params": x_decoder_zinb_params
        }

