import torch
import torch.nn as nn

from .VAEEncoder import VAEEncoder
from .ModalityPredictor import ModalityPredictor
from .CrossAttention import ModalityCrossAttention
from .CrossAttention import LearnedTempCellTypeCrossAttention
from .MultiModalDecoder import SharedDecoder

import itertools
from typing import List

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
        n_latent: int,
        batch_level_sizes: List[int],
        n_cell_type_experts: int = 10,
        encoder_hidden_dims: list = [128, 64],
        predictor_hidden_dims: list = [256, 128, 64],
        decoder_hidden_dims: list = [64, 128, 256],
    ):
        super().__init__()

        self.batch_level_sizes = batch_level_sizes

        self.n_batch = sum(batch_level_sizes)

        self.n_modalities = batch_level_sizes[0]
        self.n_studies = batch_level_sizes[1]
        self.n_samples = batch_level_sizes[2]


        # 1) Modality Experts
        self.modality_experts = nn.ModuleList([
            VAEEncoder(n_input, n_latent, encoder_hidden_dims)
            for _ in range(self.n_modalities)
        ])

        # 2) Modality predictors and discriminators and batch discriminators
        self.modality_predictor = ModalityPredictor(n_latent, self.n_modalities, predictor_hidden_dims)

        # TODO: Remove this change
        self.batch_discriminator_celltype_list = nn.ModuleList([
            ModalityPredictor(n_latent, d, predictor_hidden_dims)
            for d in self.batch_level_sizes
        ])

        # TODO: Remove this change
        self.batch_discriminator_final_list = nn.ModuleList([
            ModalityPredictor(n_latent, d, predictor_hidden_dims)
            for d in self.batch_level_sizes
        ])

        # 3) Cell Type Experts
        self.celltype_experts = nn.ModuleList([
            VAEEncoder(n_input, n_latent, encoder_hidden_dims)
            for _ in range(n_cell_type_experts)
        ])

        # 4) Cross-Attention modules
        self.modality_cross_attention = ModalityCrossAttention(n_latent)
        self.celltype_cross_attention = LearnedTempCellTypeCrossAttention(
                    latent_dim=n_latent,
                    num_modalities=self.n_modalities
                )
        # 5) Shared decoder
        self.shared_decoder = SharedDecoder(
            latent_dim=n_latent,
            n_input=n_input,
            batch_level_sizes=self.batch_level_sizes,
            hidden_dims=decoder_hidden_dims        
            )
        
        self.g_params = itertools.chain(
            self.modality_experts.parameters(),
            self.celltype_experts.parameters(),
            self.modality_cross_attention.parameters(),
            self.celltype_cross_attention.parameters(),
            self.shared_decoder.parameters(),
            self.modality_predictor.parameters(),
            )
        
        self.d_params = itertools.chain(
            self.batch_discriminator_celltype_list.parameters(),
            self.batch_discriminator_final_list.parameters()
            )

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor):
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

        # ----- 1) Run each ModalityExpert (Encoders) -----
        # using modality_idx run the modality experts, only run the modality experts for the given modality_idx

        z_modality_list = []
        modality_vae_params = []
        # modality_logits_list = []

        for i in range(self.n_modalities):
            z_m, (mu_m, logvar_m) = self.modality_experts[i](x)
            z_modality_list.append(z_m)
            modality_vae_params.append(torch.stack([mu_m, logvar_m], dim=0))

        # Convert z_modality_list to tensor
        z_modality_list = torch.stack(z_modality_list, dim=1) # -> [batch_size, num_modalities, latent_dim]
        modality_vae_params = torch.stack(modality_vae_params, dim=1) # -> [2, num_modalities, batch_size, latent_dim]
        modality_vae_params = modality_vae_params.permute(2, 1, 0, 3) # -> [batch_size, num_modalities, 2, latent_dim]


        # Gather the correct entries for each batch using idx
        z_modality_refined = z_modality_list[torch.arange(x.size(0)), modality_idx]  # shape: [256, 10]
        modality_vae_params = modality_vae_params[torch.arange(x.size(0)), modality_idx] # shape: [256, 2, latent_dim]


        # Classify modality from each z
        refined_modality_logits = self.modality_predictor(z_modality_refined)

        # TODO: Remove this
        # z_modality_list = []
        # modality_vae_params = []
        # modality_logits_list = []

        # for expert in self.modality_experts:
        #     z_m, (mu_m, logvar_m) = expert(x)
        #     z_modality_list.append(z_m)
        #     modality_vae_params.append((mu_m, logvar_m))

        #     # Classify modality from each z
        #     logits_m = self.modality_predictor(z_m)
        #     modality_logits_list.append(logits_m)

        # # => [batch_size, num_modalities, latent_dim]
        # z_modality_stack = torch.stack(z_modality_list, dim=1)

        # # ----- 2) Modality Cross-Attention -----
        # z_modality_refined = self.modality_cross_attention(
        #     z_modality_stack,
        #     modality_idx
        # )
        # # => [batch_size, latent_dim]

        # (b) Also predict modality from the refined embedding
        # refined_modality_logits = self.modality_predictor(z_modality_refined)

        # ----- 3) Cell Type Experts (Encoders) -----
        z_celltype_list = []
        celltype_vae_params = []
        batch_logits_celltype_list = []

        for expert in self.celltype_experts:
            z_c, (mu_c, logvar_c) = expert(x)
            z_celltype_list.append(z_c)
            celltype_vae_params.append((mu_c, logvar_c))

            # Predict modality from celltype expert
            batch_logits_celltype_sub_list = []
            for i in range(len(self.batch_level_sizes)):
                logits_c = self.batch_discriminator_celltype_list[i](z_c)
                batch_logits_celltype_sub_list.append(logits_c)
            batch_logits_celltype_list.append(batch_logits_celltype_sub_list)


        # => [batch_size, num_cell_type_experts, latent_dim]
        z_celltype_stack = torch.stack(z_celltype_list, dim=1)

        # ----- 4) Cell-Type Cross-Attention -----
        z_cell_refined, cell_attn_weights = self.celltype_cross_attention(
            z_modality_refined,
            z_celltype_stack,
            modality_idx # needed for per-modality temperature
        )
        # => [batch_size, latent_dim]

        # (b) Also predict batch from the final embedding
        batch_logits_final_list = [
            self.batch_discriminator_final_list[i](z_cell_refined)
            for i in range(len(self.batch_level_sizes)) 
        ]

        # ----- 5) Shared Decoder -----
        # Here we decode from the final 'z_cell_refined'
        x_reconstructed, x_decoder_zinb_params = self.shared_decoder(z_cell_refined, batch_idx)

        return {
            # "z_modality_list":       z_modality_list,
            "modality_vae_params":   modality_vae_params,
            # "modality_logits_list":  modality_logits_list,
            "z_modality_refined":    z_modality_refined,

            "refined_modality_logits": refined_modality_logits, 

            "z_celltype_list":       z_celltype_list,
            "celltype_vae_params":   celltype_vae_params,
            "z_cell_refined":        z_cell_refined,
            "cell_attn_weights":     cell_attn_weights,

            "batch_logits_celltype_list":  batch_logits_celltype_list,
            "batch_logits_final_list":     batch_logits_final_list,

            # Shared decoder parameters:
            "x_reconstructed": x_reconstructed,
            "x_decoder_zinb_params": x_decoder_zinb_params
        }
