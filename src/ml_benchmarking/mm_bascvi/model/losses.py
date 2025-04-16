import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_benchmarking.bascvi.model.distributions import NegativeBinomial
from ml_benchmarking.mm_bascvi.model.distributions import ZeroInflatedNegativeBinomial

class MMBAscVILoss:
    """
    Handles loss calculations for MMBAscVI model.
    Separates loss logic from the trainer for cleaner code organization.
    """
    
    def __init__(self, loss_weights=None, bulk_id=None, training_args=None):
        """
        Initialize loss calculator with configuration parameters.
        
        Args:
            loss_weights: Dictionary of loss component weights
            bulk_id: ID for bulk RNA-seq modality (if any)
            training_args: Additional training arguments that affect loss calculation
        """
        self.loss_weights = loss_weights or {}
        self.bulk_id = bulk_id
        self.training_args = training_args or {}
        self.batch_dict_keys = ['modality', 'study', 'sample']
        
        # Warmup parameters
        self.n_steps_kl_ramp = training_args.get("n_steps_kl_ramp", 0.0) if training_args else 0.0
        self.n_steps_adv_ramp = training_args.get("n_steps_adv_ramp", 0.0) if training_args else 0.0
        self.n_steps_adv_start = training_args.get("n_steps_adv_start", 0.0) if training_args else 0.0
    
    def is_in_warmup(self, global_step):
        """Helper to check if we're in warm-up phase"""
        return global_step < self.n_steps_adv_start
    
    def get_loss_weights(self, global_step):
        """Calculate dynamic loss weights based on training progress"""
        base_weights = self.training_args.get("loss_weights", {})

        # Default weights if not specified
        weights = {
            "reconstruction": 1000.0,
            "kl": 1.0,
            "ct_diversity": 1.0,
            "ct_regularization": 1.0,
            "adversarial": 1.0
        }

        # Use defaults for any missing weights
        weights.update(base_weights)
        
        # Apply KL ramp if configured
        if self.n_steps_kl_ramp > 0 and global_step < self.n_steps_kl_ramp:
            ramp_factor = global_step / self.n_steps_kl_ramp
            weights["kl"] *= ramp_factor
        
        # Apply adversarial weight adjustment
        if global_step < self.n_steps_adv_start:
            weights["adversarial"] = 0.0
        elif global_step < self.n_steps_adv_start + self.n_steps_adv_ramp:
            ramp_factor = (global_step - self.n_steps_adv_start) / self.n_steps_adv_ramp
            weights["adversarial"] *= ramp_factor
        
        return weights
    
    def compute_reconstruction_loss(self, x_decoder_zinb_params, x, feature_presence_mask, modality_idx) -> torch.Tensor:
        """Compute reconstruction loss between x_reconstructed and x"""
        px_rate, px_r, px_dropout = x_decoder_zinb_params

        # compute log_prob for bulk and non-bulk
        log_prob = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x)
        
        if self.bulk_id is not None:
            bulk_mask = (modality_idx == self.bulk_id)

            log_prob_bulk = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x)

            # weight the bulk loss by factor from config
            if self.training_args and "loss_weights" in self.training_args:
                bulk_weight = self.training_args["loss_weights"].get("bulk_reconstruction", 0.0)
                log_prob_bulk = log_prob_bulk * bulk_weight

            # reshape bulk_mask to match log_prob shape
            expanded_bulk_mask = bulk_mask.unsqueeze(-1).expand(-1, log_prob.size(1))

            log_prob = torch.where(expanded_bulk_mask, log_prob_bulk, log_prob)

        return torch.mean(log_prob * feature_presence_mask)

    def compute_vae_kl_loss(self, mod_vae_params, cell_vae_params):
        """Compute KL divergence loss for both modality and cell-type VAEs"""
        mod_kl_loss = 0.0
        # Sum across all modality experts
        # mod_vae_params shape: [batch_size, 2, latent_dim]
        mu = mod_vae_params[:, 0, :].squeeze(1)
        logvar = mod_vae_params[:, 1, :].squeeze(1)
        kl = 0.5 * torch.sum(logvar.exp() + mu**2 - 1.0 - logvar, dim=1).mean()
        mod_kl_loss += kl

        cell_kl_loss = 0.0
        # Sum across all cell-type experts
        for (mu, logvar) in cell_vae_params:
            kl = 0.5 * torch.sum(logvar.exp() + mu**2 - 1.0 - logvar, dim=1).mean()
            cell_kl_loss += kl

        kl_loss = (mod_kl_loss + cell_kl_loss) / (len(cell_vae_params) + 1)

        return kl_loss
    
    def compute_batch_classification_loss(self, batch_logits, batch_idx):
        """Compute batch classification loss"""
        return nn.functional.cross_entropy(batch_logits, batch_idx)
    
    def compute_batch_discriminator_accuracy(self, batch_logits, batch_idx):
        """Compute batch classification accuracy from BatchDiscriminator"""
        return (batch_logits.argmax(dim=1) == batch_idx).float().mean()
    
    def compute_celltype_diversity_loss(self, z_celltype_list):
        """Modified to penalize similarity between experts"""
        # Convert list to tensor if it's a list of celltype experts
        if isinstance(z_celltype_list, list):
            z_celltype_tensor = torch.stack(z_celltype_list, dim=1)
        else:
            z_celltype_tensor = z_celltype_list
        
        # Get dimensions
        batch_size, num_celltypes, latent_dim = z_celltype_tensor.shape
        
        # Normalize each embedding vector
        z_norm = F.normalize(z_celltype_tensor, p=2, dim=2)
        
        # Reshape to prepare for batch matrix multiplication
        # [batch_size, num_celltypes, latent_dim] -> [batch_size, num_celltypes, latent_dim]
        z_a = z_norm
        # [batch_size, num_celltypes, latent_dim] -> [batch_size, latent_dim, num_celltypes]
        z_b = z_norm.transpose(1, 2)
        
        # Batch matrix multiplication to get cosine similarities for all samples at once
        # [batch_size, num_celltypes, num_celltypes]
        cosine_sim = torch.bmm(z_a, z_b)
        
        # Create identity matrix and expand to batch size
        identity = torch.eye(num_celltypes, device=cosine_sim.device)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create mask for off-diagonal elements
        mask = torch.ones_like(identity) - identity
        
        # Apply mask to get only off-diagonal similarities
        off_diag_sim = cosine_sim * mask
        
        # Square the similarities to penalize high similarity more strongly
        squared_sim = off_diag_sim ** 2
        
        # Sum across the off-diagonal elements for each sample
        sample_losses = squared_sim.sum(dim=(1, 2))
        
        # Average across the batch
        avg_loss = sample_losses.mean()
        
        return avg_loss

    def compute_ct_low_rank_regularization(self, z_celltype_list):
        # Stack if needed
        if isinstance(z_celltype_list, list):
            z_batch = torch.stack(z_celltype_list, dim=1)
        else:
            z_batch = z_celltype_list
        
        batch_size, num_celltypes, latent_dim = z_batch.shape
        desired_rank = self.training_args.get("desired_ct_rank", 2) if self.training_args else 2
        total_rank_loss = 0.0
        
        for ct_idx in range(num_celltypes):
            z_ct = z_batch[:, ct_idx, :]
            z_centered = z_ct - z_ct.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix (much faster than SVD)
            cov = torch.matmul(z_centered.t(), z_centered) / (batch_size - 1)
            
            # Get eigenvalues (faster than full SVD)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.flip(0)  # Sort descending
            
            # Penalize eigenvalues after desired rank
            ct_rank_loss = torch.sum(eigenvalues[desired_rank:])
            total_rank_loss += ct_rank_loss
        
        return total_rank_loss / num_celltypes
    
    def compute_loss(self, outputs, batch, optimizer_idx=0, global_step=0):
        """Combined loss function for MMBAscVI model"""
        # 1) Unpack needed model outputs
        mod_vae_params = outputs["modality_vae_params"]

        z_celltype_list = outputs["z_celltype_list"]
        cell_vae_params = outputs["celltype_vae_params"]

        disc_ct_logits_list = outputs["disc_ct_logits_list"]
        disc_z_logits_list = outputs["disc_z_logits_list"]

        x_decoder_zinb_params = outputs["x_decoder_zinb_params"]
        
        # Extract batch data
        x = batch["x"]
        feature_presence_mask = batch["feature_presence_mask"]

        modality_idx = batch["batch_idx"][:, 0]
        study_idx = batch["batch_idx"][:, 1]
        sample_idx = batch["batch_idx"][:, 2]
        batch_idx_dict = {"modality": modality_idx, "study": study_idx, "sample": sample_idx}

        # Get weights from training args
        loss_weights = self.get_loss_weights(global_step)

        loss_components_dict = {}

        # Generator Step
        if optimizer_idx == 0:
            # Compute losses

            # Reconstruction Loss
            loss_components_dict["reconstruction"] = self.compute_reconstruction_loss(x_decoder_zinb_params, x, feature_presence_mask, modality_idx)

            # KL on Modality and Celltype VAEs 
            loss_components_dict["kl"] = self.compute_vae_kl_loss(mod_vae_params, cell_vae_params)

            # Celltype Diversity
            loss_components_dict["ct_diversity"] = self.compute_celltype_diversity_loss(z_celltype_list)

            # Celltype Low Rank Regularization
            loss_components_dict["ct_regularization"] = self.compute_ct_low_rank_regularization(z_celltype_list)

            total_loss = (
                loss_weights["reconstruction"] * loss_components_dict["reconstruction"] + 
                loss_weights["kl"] * loss_components_dict["kl"] +
                loss_weights["ct_diversity"] * loss_components_dict["ct_diversity"] +
                loss_weights["ct_regularization"] * loss_components_dict["ct_regularization"]
            )
            
            # Adversarial Step
            if not self.is_in_warmup(global_step):
                # Batch Discriminator Celltype
                temp_disc_ct_loss_dict = {"modality": 0.0, "study": 0.0, "sample": 0.0}
                for i in range(len(batch_idx_dict.keys())):             # iterate over batch levels
                    for j in range(len(disc_ct_logits_list)):            # iterate over celltype experts
                        temp_disc_ct_loss_dict[self.batch_dict_keys[i]] += self.compute_batch_classification_loss(disc_ct_logits_list[j][i], batch_idx_dict[self.batch_dict_keys[i]])
                    # Mean over celltype experts and add to loss components dict
                    loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"] = temp_disc_ct_loss_dict[self.batch_dict_keys[i]] / len(disc_ct_logits_list)
                
                # Batch Discriminator Final Embedding
                for i in range(len(batch_idx_dict.keys())):
                    loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"] = self.compute_batch_classification_loss(disc_z_logits_list[i], batch_idx_dict[self.batch_dict_keys[i]])

                # Discriminator Loss (negative)
                for i in range(len(self.batch_dict_keys)):
                    total_loss -= loss_weights["adversarial"] * self.training_args["loss_weights"]["ct_discriminator"][i] * loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"] 
                    total_loss -= loss_weights["adversarial"] * self.training_args["loss_weights"]["z_discriminator"][i] * loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"] 

            
        # Discriminator Step
        elif optimizer_idx == 1:

            # Batch Discriminator Celltype
            temp_disc_ct_loss_dict = {"modality": 0.0, "study": 0.0, "sample": 0.0}
            for i in range(len(batch_idx_dict.keys())):             # iterate over batch levels
                for j in range(len(disc_ct_logits_list)):            # iterate over celltype experts
                    temp_disc_ct_loss_dict[self.batch_dict_keys[i]] += self.compute_batch_classification_loss(disc_ct_logits_list[j][i], batch_idx_dict[self.batch_dict_keys[i]])
                # Mean over celltype experts and add to loss components dict
                loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"] = temp_disc_ct_loss_dict[self.batch_dict_keys[i]] / len(disc_ct_logits_list)
            

            # Batch Discriminator Final Embedding
            for i in range(len(batch_idx_dict.keys())):
                loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"] = self.compute_batch_classification_loss(disc_z_logits_list[i], batch_idx_dict[self.batch_dict_keys[i]])


            # Discriminator Accuracy
            temp_disc_ct_loss_dict = {"modality": 0.0, "study": 0.0, "sample": 0.0}
            for i in range(len(batch_idx_dict.keys())):             # iterate over batch levels
                for j in range(len(disc_ct_logits_list)):            # iterate over celltype experts
                    temp_disc_ct_loss_dict[self.batch_dict_keys[i]] += self.compute_batch_discriminator_accuracy(disc_ct_logits_list[j][i], batch_idx_dict[self.batch_dict_keys[i]])
                # Mean over celltype experts and add to loss components dict
                loss_components_dict[f"acc/disc_ct_{self.batch_dict_keys[i]}"] = temp_disc_ct_loss_dict[self.batch_dict_keys[i]] / len(disc_ct_logits_list)
            
            for i in range(len(batch_idx_dict.keys())):
                loss_components_dict[f"acc/disc_z_{self.batch_dict_keys[i]}"] = self.compute_batch_discriminator_accuracy(disc_z_logits_list[i], batch_idx_dict[self.batch_dict_keys[i]])

            # Final Discriminator Loss (positive)
            total_loss = 0.0
            for i in range(len(self.batch_dict_keys)):
                if self.training_args and "loss_weights" in self.training_args:
                    ct_disc_weight = self.training_args["loss_weights"].get("ct_discriminator", [1.0, 1.0, 1.0])[i]
                    z_disc_weight = self.training_args["loss_weights"].get("z_discriminator", [1.0, 1.0, 1.0])[i]
                    total_loss += ct_disc_weight * loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"]
                    total_loss += z_disc_weight * loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"]
                else:
                    total_loss += loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"]
                    total_loss += loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"]

        return total_loss, loss_components_dict