import torch
import torch.nn as nn

from .VAEEncoder import VAEEncoder
from .ModalityPredictor import ModalityPredictor
from .CrossAttention import ModalityCrossAttention
from .CrossAttention import LearnedTempCellTypeCrossAttention
from .MultiModalDecoder import SharedDecoder

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
        n_genes: int,
        n_latent: int,
        n_modalities: int,
        num_cell_type_experts: int = 10,
        encoder_hidden_dims: list = [128, 64],
        predictor_hidden_dims: list = [128, 64],
        decoder_hidden_dims: list = [128, 64],
    ):
        super().__init__()

        # 1) Modality Experts
        self.modality_experts = nn.ModuleList([
            VAEEncoder(n_genes, n_latent, encoder_hidden_dims)
            for _ in range(n_modalities)
        ])

        # 2) Modality Predictors (for classification of z)
        self.modality_predictor = ModalityPredictor(n_latent, n_modalities, predictor_hidden_dims)
        self.modality_predictor_celltype = ModalityPredictor(n_latent, n_modalities, predictor_hidden_dims)

        # 3) Cell Type Experts
        self.celltype_experts = nn.ModuleList([
            VAEEncoder(n_genes, n_latent, encoder_hidden_dims)
            for _ in range(num_cell_type_experts)
        ])

        # 4) Cross-Attention modules
        self.modality_cross_attention = ModalityCrossAttention(n_latent)
        self.celltype_cross_attention = LearnedTempCellTypeCrossAttention(
                    latent_dim=n_latent,
                    num_modalities=n_modalities
                )
        # 5) Shared decoder
        self.shared_decoder = SharedDecoder(
            latent_dim=n_latent,
            n_genes=n_genes,
            num_modalities=n_modalities,
            hidden_dims=decoder_hidden_dims        
            )

    def forward(self, x: torch.Tensor, modality_vec: torch.Tensor):
        """
        Args:
            x:           [batch_size, n_genes]
            modality_vec:[batch_size] with values in {0,1,2} for 3 modalities
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

        # ----- 1) Run each ModalityExpert (Encoders) -----
        z_modality_list = []
        modality_vae_params = []
        modality_logits_list = []

        for expert in self.modality_experts:
            z_m, (mu_m, logvar_m) = expert(x)
            z_modality_list.append(z_m)
            modality_vae_params.append((mu_m, logvar_m))

            # Classify modality from each z
            logits_m = self.modality_predictor(z_m)
            modality_logits_list.append(logits_m)

        # => [batch_size, num_modalities, latent_dim]
        z_modality_stack = torch.stack(z_modality_list, dim=1)

        # ----- 2) Modality Cross-Attention -----
        z_modality_refined = self.modality_cross_attention(
            z_modality_stack,
            modality_vec
        )
        # => [batch_size, latent_dim]

        # (b) Also predict modality from the refined embedding
        refined_modality_logits = self.modality_predictor(z_modality_refined)


        # ----- 3) Cell Type Experts (Encoders) -----
        z_celltype_list = []
        celltype_vae_params = []
        celltype_logits_list = []

        for expert in self.celltype_experts:
            z_c, (mu_c, logvar_c) = expert(x)
            z_celltype_list.append(z_c)
            celltype_vae_params.append((mu_c, logvar_c))

            # Predict modality from celltype expert
            logits_c = self.modality_predictor_celltype(z_c)
            celltype_logits_list.append(logits_c)

        # => [batch_size, num_cell_type_experts, latent_dim]
        z_celltype_stack = torch.stack(z_celltype_list, dim=1)

        # ----- 4) Cell-Type Cross-Attention -----
        z_cell_refined, cell_attn_weights = self.celltype_cross_attention(
            z_modality_refined,
            z_celltype_stack,
            modality_vec # needed for per-modality temperature
        )
        # => [batch_size, latent_dim]

        # ----- 5) Shared Decoder -----
        # Here we decode from the final 'z_modality_refined'
        x_reconstructed, x_decoder_zinb_params = self.shared_decoder(z_modality_refined, modality_vec)

        return {
            "z_modality_list":       z_modality_list,
            "modality_vae_params":   modality_vae_params,
            "modality_logits_list":  modality_logits_list,
            "z_modality_refined":    z_modality_refined,

            "refined_modality_logits": refined_modality_logits, 


            "z_celltype_list":       z_celltype_list,
            "celltype_vae_params":   celltype_vae_params,
            "celltype_logits_list":  celltype_logits_list,
            "z_cell_refined":        z_cell_refined,
            "cell_attn_weights":     cell_attn_weights,

            # Shared decoder parameters:
            "x_reconstructed": x_reconstructed,
            "x_decoder_zinb_params": x_decoder_zinb_params
        }
