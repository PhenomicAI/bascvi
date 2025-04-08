import torch
import torch.nn as nn

class ModalityCrossAttention(nn.Module):
    """
    For each sample:
      - Query is the embedding of the sample's *true* modality
      - Key/Value = all modality embeddings (scrna, spatial, bulk, etc.)
    Outputs a 'refined' modality embedding.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=1, 
            batch_first=True
        )
        self.ln = nn.LayerNorm(latent_dim)

    def forward(
        self, 
        z_modality_stack: torch.Tensor,   # [batch_size, num_modalities, latent_dim]
        modality_idx: torch.Tensor        # [batch_size]
    ) -> torch.Tensor:
        bsz, num_mods, dim = z_modality_stack.shape

        # Convert modality_idx to long type
        modality_idx = modality_idx.long()  
        
        # Gather queries for each sample
        query_list = []
        for i in range(bsz):
            query_list.append(z_modality_stack[i, modality_idx[i], :].unsqueeze(0))
        # [batch_size, 1, latent_dim]
        query = torch.cat(query_list, dim=0)
        query = query.unsqueeze(1)

        # Single pass of multihead attention
        refined, _ = self.attn(query, z_modality_stack, z_modality_stack)
        refined = refined.squeeze(1)  # -> [batch_size, latent_dim]

        # Residual connection with the original query
        # shape of query.squeeze(1): [batch_size, latent_dim]
        residual = query.squeeze(1)
        
        # Direct addition
        refined = refined + residual

        # Layer norm
        refined = self.ln(refined)

        return refined


import math
import torch
import torch.nn as nn

class LearnedTempCellTypeCrossAttention(nn.Module):
    """
    Vectorized single-head attention that learns a separate temperature per modality.
    
    For each sample i in the batch:
      - We look up T_m = exp(log_temps[ modality_idx[i] ])
      - We divide that sample’s attention scores by T_m
      - Then apply a softmax across the cell-type dimension.
    
    This way, single-cell domain can learn a lower temperature (sharper distribution),
    while bulk/spatial can learn a higher temperature (softer mixture).
    """
    def __init__(self, latent_dim: int, ct_latent_dim: int, num_modalities: int, residual_weight: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.ct_latent_dim = ct_latent_dim
        self.num_modalities = num_modalities
        self.residual_weight = residual_weight

        # Simple linear layers to project query/key/value
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj   = nn.Linear(ct_latent_dim, latent_dim)
        self.value_proj = nn.Linear(ct_latent_dim, latent_dim)

        # Learnable log-temperatures for each modality, initialized at 0 => exp(0)=1.
        # shape: [num_modalities]
        self.log_temps = nn.Parameter(torch.zeros(num_modalities))

        # For dot-product scaling
        self.scale_factor = math.sqrt(latent_dim)

        # Layer norm
        self.ln = nn.LayerNorm(latent_dim)


    def forward(
        self,
        z_modality_refined: torch.Tensor,  # [B, latent_dim]
        z_celltype_stack: torch.Tensor,    # [B, num_cell_type_experts, latent_dim]
        modality_idx: torch.Tensor         # [B] in {0, ..., num_modalities-1}
    ):
        """
        Args:
          z_modality_refined: [B, latent_dim]          (query)
          z_celltype_stack:   [B, num_cell_type_experts, latent_dim]  (keys/values)
          modality_idx:       [B] each entry in [0..num_modalities-1]
        
        Returns:
          attn_output:  [B, latent_dim]  refined embedding after cross-attention
          attn_weights: [B, num_cell_type_experts] attention distribution over cell-type experts
        """
        # Convert modality_idx to long type
        modality_idx = modality_idx.long()  


        B, latent_dim = z_modality_refined.shape
        _, num_ct, _  = z_celltype_stack.shape

        # Project to query/key/value spaces
        # q: [B, latent_dim]
        # k,v: [B, num_ct, latent_dim]
        q = self.query_proj(z_modality_refined)       # [B, latent_dim]
        k = self.key_proj(z_celltype_stack)           # [B, num_ct, latent_dim]
        v = self.value_proj(z_celltype_stack)         # [B, num_ct, latent_dim]

        # -- 1) Compute Raw Scores (batch-wise) --
        # We want a batched matrix multiply: (B, 1, d) x (B, d, num_ct) => (B, 1, num_ct)
        # So expand q => [B, 1, d], transpose k => [B, d, num_ct]
        q_reshaped = q.unsqueeze(1)             # [B, 1, d]
        k_trans    = k.transpose(1, 2)          # [B, d, num_ct]
        scores     = torch.bmm(q_reshaped, k_trans)  # [B, 1, num_ct]
        scores     = scores / self.scale_factor       # scale by sqrt(latent_dim)

        # -- 2) Apply the per-sample temperature --
        # log_temps => shape [num_modalities]
        # gather temperatures for each sample => shape [B]
        sample_log_temps = self.log_temps[modality_idx]   # [B]
        sample_temps     = sample_log_temps.exp()         # [B]

        # Add annealing factor based on global step (new)
        min_temp = 0.1  # Minimum temperature
        max_temp = 1.0  # Maximum temperature
        
        # Allow cross-attention to become sharper over time
        if hasattr(self, 'current_epoch'):
            annealing_factor = min(1.0, self.current_epoch / 10)  # Anneal over 10 epochs
            effective_temp = min_temp + (max_temp - min_temp) * (1 - annealing_factor)
            sample_temps = sample_temps * effective_temp
        
        # Expand so we can divide the [B, 1, num_ct] scores
        # shape needed => [B, 1, 1]
        sample_temps = sample_temps.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        scores = scores / sample_temps                         # broadcast across num_ct

        # -- 3) Softmax along the cell-type dimension --
        attn_weights = torch.softmax(scores, dim=2)  # [B, 1, num_ct]

        # -- 4) Weighted sum of values => [B, 1, d]
        # v: [B, num_ct, d]
        refined = torch.bmm(attn_weights, v)     # [B, 1, d]

        # Squeeze extra dimensions
        refined  = refined.squeeze(1)   # [B, d]
        attn_weights = attn_weights.squeeze(1)  # [B, num_ct]

        # -- 5) Residual connection with the original query
        # shape of z_modality_refined: [batch_size, latent_dim]
        residual = z_modality_refined

        # Direct addition of the residual
        refined = refined + residual * self.residual_weight

        # Layer norm
        refined = self.ln(refined)

        return refined, attn_weights
