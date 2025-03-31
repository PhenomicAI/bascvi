# train_lightning.py

import os
import shutil
import logging
import itertools

import wandb
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.utils as utils
import torch.nn.functional as F
from torch import nn, optim

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.model.distributions import NegativeBinomial
from ml_benchmarking.bascvi.utils.utils import umap_calc_and_save_html, calc_kni_score, calc_rbni_score
from ml_benchmarking.mm_bascvi.model.distributions import ZeroInflatedNegativeBinomial


from ml_benchmarking.mm_bascvi.model.MMBAscVI import MMBAscVI  # Adjust import to your structure

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MMBAscVITrainer(pl.LightningModule):
    """
    Wraps the MMBAscVI model in a LightningModule, handling:
      - forward pass
      - loss computation in training_step
      - optimizer configuration
      - validation and prediction
      - wandb integration
    """

    def __init__(
        self,
        root_dir,
        soma_experiment_uri: str = None,
        gene_list: list = None,
        model_args: dict = {},
        training_args: dict = {},
        callbacks_args: dict = {},
        modalities_idx_to_name_dict: dict = {},
    ):
        super().__init__()
        # Save hyperparameters in hparams.yaml file
        self.save_hyperparameters(ignore="datamodule")
        
        self.root_dir = root_dir
        self.valid_counter = 0

        self.gene_list = gene_list
        
        self.soma_experiment_uri = soma_experiment_uri
        self.obs_df = None
        
        self.model_args = model_args
        self.training_args = training_args
        self.callbacks_args = callbacks_args
        
        # Get specific training parameters
        self.kl_loss_weight = training_args.get("kl_loss_weight", 1.0)
        self.n_epochs_kl_warmup = training_args.get("n_epochs_kl_warmup")
        self.n_steps_kl_warmup = training_args.get("n_steps_kl_warmup")
        
        self.save_validation_umaps = training_args.get("save_validation_umaps", False)
        self.run_validation_metrics = training_args.get("run_validation_metrics", False)
        self.save_validation_embeddings = training_args.get("save_validation_embeddings", False)

        self.modalities_idx_to_name_dict = modalities_idx_to_name_dict
        
        self.model = MMBAscVI(**self.model_args)
        
        # Configure callbacks
        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize MMBAscVI Trainer")
        
        self.automatic_optimization = False  # PyTorch Lightning 2 method
        
        self.validation_z_cell_refined = []
        self.validation_z_cell_list = []
        
        # Create validation_umaps directory if needed
        if os.path.isdir(os.path.join(self.root_dir, "validation_umaps")):
            shutil.rmtree(os.path.join(self.root_dir, "validation_umaps"))

    def configure_callbacks(self):
        """Configure callbacks for training (checkpointing, early stopping)"""
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.callbacks_args["model_checkpoint"]["monitor"],
            mode=self.callbacks_args["model_checkpoint"]["mode"],
            filename="mmbascvi-{epoch:02d}-{val_loss:.2f}"
        )
        self.callbacks.append(self.checkpoint_callback)
        
        if self.callbacks_args.get("early_stopping"):
            self.early_stop_callback = EarlyStopping(
                monitor=self.callbacks_args["early_stopping"]["monitor"],
                min_delta=0.00,
                verbose=True,
                patience=self.callbacks_args["early_stopping"]["patience"],
                mode=self.callbacks_args["early_stopping"]["mode"]
            )
            self.callbacks.append(self.early_stop_callback)

    def forward(self, batch):
        """Forward pass of the underlying MMBAscVI model."""
        x = batch["x"]
        batch_idx = batch["batch_idx"]
        return self.model(x, batch_idx)
    
    @property
    def kl_warmup_weight(self):
        """Scaling factor on KL divergence during training."""
        epoch_criterion = self.n_epochs_kl_warmup is not None
        step_criterion = self.n_steps_kl_warmup is not None
        cyclic_criterion = self.training_args.get("cyclic_kl_period", False)

        if epoch_criterion:
            kl_weight = min(1.0, self.current_epoch / self.n_epochs_kl_warmup)
        elif step_criterion:
            kl_weight = min(1.0, self.global_step / self.n_steps_kl_warmup)
        elif cyclic_criterion:
            kl_weight = self._get_kld_cycle(self.current_epoch, period=cyclic_criterion)
        else:
            kl_weight = 1.0

        return kl_weight
    
    def _get_kld_cycle(self, epoch, period=20):
        '''
        0-10: 0 to 1
        10-20 1
        21-30 0 to 1
        30-40 1
        '''
        ct = epoch % period
        pt = epoch % (period//2)
        if ct >= period//2:
            return 1
        else:
            return min(1, (pt) / (period//2))
    
    def compute_reconstruction_loss(self, x_decoder_zinb_params, x, feature_presence_mask, modality_idx) -> torch.Tensor:
        """Compute reconstruction loss between x_reconstructed and x"""

        px_rate, px_r, px_dropout = x_decoder_zinb_params

        # search for "Bulk" in modalities_idx_to_name_dict values, get key
        bulk_id = next((key for key, value in self.modalities_idx_to_name_dict.items() if "Bulk" in value), None)

        # compute log_prob for bulk and non-bulk
        log_prob = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x)
        
        if bulk_id is not None:
            bulk_mask = (modality_idx == bulk_id)

            log_prob_bulk = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x)

            # weight the bulk loss by 0
            log_prob_bulk = log_prob_bulk * 0.0 # TODO: remove this

            # reshape bulk_mask to match log_prob shape
            expanded_bulk_mask = bulk_mask.unsqueeze(-1).expand(-1, log_prob.size(1))

            log_prob = torch.where(expanded_bulk_mask, log_prob_bulk, log_prob)

        return torch.mean(log_prob * feature_presence_mask)
    
    # def compute_deconvolution_loss(self, z_modality_refined, cell_attn_weights, z_celltype_list) -> torch.Tensor:
    #     """
    #     Computes the deconvolution loss:
    #     L_deconv = || z_modality_refined - sum_j(cell_attn_weights * z_celltype_list) ||^2
    #     """
    #     # Stack the cell-type embeddings: [batch_size, num_ct, latent_dim]
    #     z_cell_stack = torch.stack(z_celltype_list, dim=1)  # [batch_size, num_ct, latent_dim]

    #     # Expand the attention weights to match the cell-type embedding dimensions
    #     alpha_expanded = cell_attn_weights.unsqueeze(-1)  # [batch_size, num_ct, 1]

    #     # Compute the weighted sum of cell-type embeddings
    #     celltype_combined = (alpha_expanded * z_cell_stack).sum(dim=1)  # [batch_size, latent_dim]

    #     # Compute the L2 difference between the refined modality embedding and the weighted sum
    #     l2_per_sample = (z_modality_refined - celltype_combined).pow(2).sum(dim=1)

    #     # Compute the mean loss across all samples
    #     deconv_loss = l2_per_sample.mean() 

    #     return deconv_loss

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
        #cell_kl_loss = cell_kl_loss / len(cell_vae_params)

        kl_loss = (mod_kl_loss + cell_kl_loss) / (len(cell_vae_params) + 1)
        return kl_loss * self.kl_warmup_weight * self.kl_loss_weight

    def compute_modality_classification_loss(self, mod_logits_list, refined_mod_logits, modality_idx):
        """Compute modality classification loss from ModalityExperts"""
        # ce_modality_raw = 0.0
        # for logits_m in mod_logits_list:
        #     ce_modality_raw += F.cross_entropy(logits_m, modality_idx)

        ce_modality_refined = F.cross_entropy(refined_mod_logits, modality_idx)

        return ce_modality_refined #ce_modality_raw # + 0.5 * ce_modality_refined
    
    def compute_batch_discriminator_loss(self, batch_logits, batch_idx):
        """Compute batch classification loss from BatchDiscriminator"""
        return nn.functional.cross_entropy(batch_logits, batch_idx)
    
    def compute_batch_discriminator_accuracy(self, batch_logits, batch_idx):
        """Compute batch classification accuracy from BatchDiscriminator"""
        return (batch_logits.argmax(dim=1) == batch_idx).float().mean()
    
    def compute_celltype_diversity_loss(self, z_celltype_list):
        """Vectorized sample-level celltype diversity loss using cosine similarity"""
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
        
        # Take absolute values to handle potential negative similarities
        abs_sim = torch.abs(off_diag_sim)
        
        # Sum across the off-diagonal elements for each sample
        # [batch_size]
        sample_losses = abs_sim.sum(dim=(1, 2))
        
        # Average across the batch
        avg_loss = sample_losses.mean()
        
        return avg_loss

    def compute_celltype_diversity_loss_OLD(self, z_celltype_list):
        """Celltype diversity loss: Minimize off-diagonal similarity; keep cell-type embeddings distinct"""
        # Convert list to tensor if it's a list of celltype experts
        if isinstance(z_celltype_list, list):
            z_celltype_tensor = torch.stack(z_celltype_list, dim=1)
        else:
            z_celltype_tensor = z_celltype_list
            
        # z_celltype_tensor shape: [batch_size, num_cell_type_experts, latent_dim]
        
        # Normalize embeddings along the latent dimension
        z_norm = F.normalize(z_celltype_tensor, p=2, dim=2)
        # z_norm shape: [batch_size, num_cell_type_experts, latent_dim]
        
        # Compute similarity matrix between cell type experts
        # First transpose to [batch_size, latent_dim, num_cell_type_experts] * [batch_size, num_cell_type_experts, latent_dim]
        # Then multiply to get [batch_size, num_cell_type_experts, num_cell_type_experts]
        similarity_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))
        
        # Average over batch dimension
        similarity_matrix = similarity_matrix.mean(dim=0)

        # Create an identity matrix for the number of cell type experts
        num_experts = similarity_matrix.shape[0]
        identity = torch.eye(num_experts, device=similarity_matrix.device)
        
        # Compute the mean off-diagonal similarity
        off_diagonal = similarity_matrix * (1 - identity)

        # Use mean squared value instead of absolute sum
        diversity_loss = torch.mean(off_diagonal ** 2)

        # Add a small epsilon to prevent numerical issues
        return diversity_loss + 1e-8
        
    def compute_loss(self, outputs, batch, optimizer_idx=0):
        """Combined loss function for MMBAscVI model"""
        # 1) Unpack model outputs
        # z_mod_list = outputs["z_modality_list"]
        mod_vae_params = outputs["modality_vae_params"]

        # mod_logits_list = outputs["modality_logits_list"]
        refined_mod_logits = outputs["refined_modality_logits"]

        z_modality_refined = outputs["z_modality_refined"]
        cell_attn_weights = outputs["cell_attn_weights"]

        z_celltype_list = outputs["z_celltype_list"]
        cell_vae_params = outputs["celltype_vae_params"]

        batch_logits_celltype_list = outputs["batch_logits_celltype_list"]
        batch_logits_final_list = outputs["batch_logits_final_list"]

        x_decoder_zinb_params = outputs["x_decoder_zinb_params"]
        
        # Extract batch data
        x = batch["x"]
        feature_presence_mask = batch["feature_presence_mask"]

        modality_idx = batch["batch_idx"][:, 0]
        study_idx = batch["batch_idx"][:, 1]
        sample_idx = batch["batch_idx"][:, 2]
        batch_idx_dict = {"modality": modality_idx, "study": study_idx, "sample": sample_idx}
        batch_dict_keys = ['modality', 'study', 'sample']

        # Get weights from training args
        reconstruction_loss_weight = self.training_args.get("reconstruction_loss_weight", 1.0)
        kl_loss_weight = self.training_args.get("kl_loss_weight", 1.0)

        mod_class_weight = self.training_args.get("modality_class_weight", 1.0)

        batch_discriminator_ct_weight = self.training_args.get("batch_discriminator_ct_weight", 1.0)
        batch_discriminator_final_weight = self.training_args.get("batch_discriminator_final_weight", 1.0)
        # deconv_weight = self.training_args.get("deconv_weight", 1.0)
        diversity_weight = self.training_args.get("diversity_weight", 1.0)
        
        # Generator
        if optimizer_idx == 0:
            # Compute losses

            # Reconstruction Loss
            reconstruction_loss = self.compute_reconstruction_loss(x_decoder_zinb_params, x, feature_presence_mask, modality_idx)

            # Modality VAEs 
            kl_loss = self.compute_vae_kl_loss(mod_vae_params, cell_vae_params)
            
            # Modality Classification
            modality_class_loss = self.compute_modality_classification_loss(
                None, refined_mod_logits, modality_idx)
            
            # print(len(batch_logits_celltype_list))
            # print(len(batch_logits_celltype_list[0]))
            # print(batch_logits_celltype_list[0][0].size())
            # print(batch_logits_celltype_list[0][0])
            # print(batch_idx_dict[list(batch_idx_dict.keys())[0]])
            # print(batch_idx_dict[list(batch_idx_dict.keys())[0]].size())

            batch_level_weights = self.training_args.get("discriminator_batch_level_weights", [1.0, 1.0, 1.0])

            # Batch Discriminator Celltype
            batch_ct_class_loss_dict = {"modality": 0.0, "study": 0.0, "sample": 0.0}
            for i in range(len(batch_logits_celltype_list)):    # iterate over celltype experts
                for j in range(len(batch_idx_dict.keys())):          # iterate over batch levels
                    batch_ct_class_loss_dict[batch_dict_keys[j]] += self.compute_batch_discriminator_loss(batch_logits_celltype_list[i][j], batch_idx_dict[batch_dict_keys[0]]) * batch_level_weights[j]
            
            # Mean over celltype experts
            batch_ct_class_loss_dict = {k: v / len(batch_logits_celltype_list) for k, v in batch_ct_class_loss_dict.items()}
            # Update key names
            batch_ct_class_loss_dict = {f"batch_ct_{k}": v for k, v in batch_ct_class_loss_dict.items()}

            # Mean over batch levels
            batch_ct_class_loss = sum(batch_ct_class_loss_dict.values()) / len(batch_idx_dict.keys())


            # Batch Discriminator Final
            batch_final_class_loss_dict = {}
            for i in range(len(batch_idx_dict.keys())):
                batch_final_class_loss_dict[batch_dict_keys[i]] = self.compute_batch_discriminator_loss(batch_logits_final_list[i], batch_idx_dict[batch_dict_keys[i]]) * batch_level_weights[i]
            batch_final_class_loss = sum(batch_final_class_loss_dict.values()) / len(batch_idx_dict.keys())

            # Update key names
            batch_final_class_loss_dict = {f"batch_final_{k}": v for k, v in batch_final_class_loss_dict.items()}
            
            # deconv_loss = self.compute_deconvolution_loss(
            #     z_modality_refined, cell_attn_weights, z_celltype_list)
            
            # Celltype Diversity
            celltype_diversity_loss = self.compute_celltype_diversity_loss(z_celltype_list)

            batch_final_class_acc = self.compute_batch_discriminator_accuracy(batch_logits_final_list[0], batch_idx_dict[batch_dict_keys[0]])


            total_loss = (
                reconstruction_loss_weight * reconstruction_loss + 
                kl_loss_weight * kl_loss + 
                mod_class_weight * modality_class_loss - 
                batch_discriminator_ct_weight * batch_ct_class_loss -
                batch_discriminator_final_weight * batch_final_class_loss +
                # deconv_weight * deconv_loss + 
                diversity_weight * celltype_diversity_loss
            )
            
            loss_dict = {
                "loss": total_loss,

                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,

                "modality_class_loss": modality_class_loss,

                "discriminator_loss": batch_discriminator_ct_weight * batch_ct_class_loss + batch_discriminator_final_weight * batch_final_class_loss,
                "batch_discriminator_ct_loss": batch_ct_class_loss,
                "batch_discriminator_final_loss": batch_final_class_loss,

                # "deconv_loss": deconv_loss,
                "celltype_diversity_loss": celltype_diversity_loss,

                "batch_final_class_acc": batch_final_class_acc
            }

            # Add sub-losses
            loss_dict.update(batch_ct_class_loss_dict)
            loss_dict.update(batch_final_class_loss_dict)
        
        # Discriminator
        elif optimizer_idx == 1:
            # Batch Discriminator Celltype
            batch_ct_class_loss_dict = {"modality": 0.0, "study": 0.0, "sample": 0.0}
            for i in range(len(batch_logits_celltype_list)):    # iterate over celltype experts
                for j in range(len(batch_idx_dict.keys())):          # iterate over batch levels
                    batch_ct_class_loss_dict[list(batch_idx_dict.keys())[j]] += self.compute_batch_discriminator_loss(batch_logits_celltype_list[i][j], batch_idx_dict[list(batch_idx_dict.keys())[j]])
            # Mean over celltype experts
            batch_ct_class_loss_dict = {k: v / len(batch_logits_celltype_list) for k, v in batch_ct_class_loss_dict.items()}
            # Mean over batch levels
            batch_ct_class_loss = sum(batch_ct_class_loss_dict.values()) / len(batch_idx_dict.keys())
            # Update key names
            batch_ct_class_loss_dict = {f"batch_ct_{k}": v for k, v in batch_ct_class_loss_dict.items()}

            # Batch Discriminator Final
            batch_final_class_loss_dict = {}
            for i in range(len(batch_idx_dict.keys())):
                batch_final_class_loss_dict[list(batch_idx_dict.keys())[i]] = self.compute_batch_discriminator_loss(batch_logits_final_list[i], batch_idx_dict[list(batch_idx_dict.keys())[i]])
            # Mean over batch levels
            batch_final_class_loss = sum(batch_final_class_loss_dict.values()) / len(batch_final_class_loss_dict.items())
            # Update key names
            batch_final_class_loss_dict = {f"batch_final_{k}": v for k, v in batch_final_class_loss_dict.items()}

            loss_dict = {
                "loss": batch_discriminator_ct_weight * batch_ct_class_loss + batch_discriminator_final_weight * batch_final_class_loss,
                "discriminator_loss": batch_discriminator_ct_weight * batch_ct_class_loss + batch_discriminator_final_weight * batch_final_class_loss,
                "batch_discriminator_ct_loss": batch_ct_class_loss,
                "batch_discriminator_final_loss": batch_final_class_loss
            }

            # Add sub-losses
            loss_dict.update(batch_ct_class_loss_dict)
            loss_dict.update(batch_final_class_loss_dict)
        
        return loss_dict

    def training_step(self, batch, batch_idx):
        """Lightning hook that runs one training step"""
        # Skip small batches
        if batch["x"].shape[0] < 3:
            return None
        
        g_opt, d_opt = self.optimizers()

        g_opt.zero_grad()
        g_outputs = self.forward(batch)
        g_losses = self.compute_loss(g_outputs, batch, optimizer_idx=0)
        self.manual_backward(g_losses['loss'])
        utils.clip_grad_norm_(self.model.parameters(), 1.0) # clip gradients
        g_opt.step()

        d_opt.zero_grad()
        d_outputs = self.forward(batch)
        d_losses = self.compute_loss(d_outputs, batch, optimizer_idx=1)
        self.manual_backward(d_losses['loss'])
        # clip gradients
        utils.clip_grad_norm_(self.model.parameters(), 1.0) # clip gradients
        d_opt.step()
    
        # Log training losses
        train_losses = {f"train_loss/{k}": v for k, v in g_losses.items()}
        self.log_dict(train_losses)
        
        # Log trainer metrics
        # self.log("trainer/kl_warmup_weight", self.kl_warmup_weight)
        self.log("trainer/global_step", self.global_step)

        # Get current learning rates        
        self.log("trainer/vae_lr", g_opt.param_groups[0]['lr'])
        self.log("trainer/discriminator_lr", d_opt.param_groups[0]['lr'])

        # Log attention temps
        log_temps_per_modality = self.model.celltype_cross_attention.log_temps
        for i, temp in enumerate(log_temps_per_modality):
            self.log(f"trainer/celltype_attn_temp_modality_{self.modalities_idx_to_name_dict[i]}", torch.exp(temp))
        
        # Log scaling factor
        for i in range(len(self.model.modality_experts)):
            self.log(f"trainer/scaling_factor_modality_{self.modalities_idx_to_name_dict[i]}", torch.exp(self.model.modality_experts[i].log_scaling_factor))
        
        return g_losses

    def validation_step(self, batch, batch_idx):
        """Lightning hook for validation step"""
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log validation losses
        val_losses = {f"val_loss/{k}": v for k, v in losses.items()}
        self.log_dict(val_losses, on_step=False, on_epoch=True)
        
        # For UMAP visualization, extract latent representations
        z_cell_refined = outputs["z_cell_refined"]
        z_cell_list = outputs["z_celltype_list"]
        
        # Make z float64 dtype to concat properly with soma_joinid
        z = z_cell_refined.double()
        
        if "soma_joinid" in batch:
            emb_output = torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)
            self.validation_z_cell_refined.append(emb_output)
            self.validation_z_cell_list.append(z_cell_list)
        return losses

    def on_validation_epoch_end(self):
        """Process validation results at the end of the validation epoch"""
        metrics_to_log = {}
        
        if self.save_validation_umaps and self.validation_z_cell_refined:
            logger.info("Running validation UMAP...")
            embeddings = torch.cat(self.validation_z_cell_refined, dim=0).double().detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

            # Celltype UMAPs
            ct_df_list = []
            # Recursively concatenate the list of tensors for cell type embeddings
            ct_embeddings_list = []
            for batch_embeddings in self.validation_z_cell_list:
                # Each batch_embeddings is a list of tensors, one for each cell type expert
                # We need to stack them to get [batch_size, num_celltype_experts, latent_dim]
                stacked_batch = torch.stack(batch_embeddings, dim=1)
                ct_embeddings_list.append(stacked_batch)
            
            # Now concatenate along the batch dimension
            ct_embeddings = torch.cat(ct_embeddings_list, dim=0).double().detach().cpu().numpy()  # [num_validation_samples, num_celltype_experts, num_latent_dims]
            num_celltype_experts = ct_embeddings.shape[1]
            num_latent_dims = ct_embeddings.shape[2]

           
           
            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            os.makedirs(save_dir, exist_ok=True)



            obs_columns = ["standard_true_celltype", "study_name", "sample_name", "scrnaseq_protocol", "tissue_collected"]
            if "species" not in obs_columns:
                obs_columns.append("species")

            if self.obs_df is None and self.soma_experiment_uri:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    self.obs_df = soma_experiment.obs.read(column_names=['soma_joinid'] + obs_columns).concat().to_pandas()
                    self.obs_df = self.obs_df.set_index("soma_joinid")
            
            if self.obs_df is not None:
                if "species" not in self.obs_df.columns:
                    logger.info("Adding species column to obs, assuming human data")
                    self.obs_df["species"] = "human"

                    # assign species based on study_name
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_m-"), "species"] = "mouse"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_r-"), "species"] = "rat"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_l-"), "species"] = "lemur"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_c-"), "species"] = "macaque"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_f-"), "species"] = "fly"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_a-"), "species"] = "axolotl"
                    self.obs_df.loc[self.obs_df["study_name"].str.contains("_z-"), "species"] = "zebrafish"
                
                embeddings_df, fig_path_dict = umap_calc_and_save_html(
                    embeddings_df.set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns, save_dir, obs_columns, max_cells=200000, opacity=max((1 - embeddings_df.shape[0] / 200000), 0.3) # more opacity for fewer cells
                ) 

                for key, fig_path in fig_path_dict.items():
                    metrics_to_log["z_cell_refined/" + key] = wandb.Image(fig_path, caption=key)


                # UMAP CT
                for i in range(num_celltype_experts):
                    ct_save_dir = os.path.join(save_dir, "celltype_umaps", f"ct_{i}")
                    os.makedirs(ct_save_dir, exist_ok=True)
                    ct_emb_columns = [f"ct_{i}_dim_{j}" for j in range(num_latent_dims)]
                    ct_df = pd.DataFrame(data=ct_embeddings[:, i, :], columns=ct_emb_columns)
                    ct_df["soma_joinid"] = embeddings_df["soma_joinid"]
                    ct_df, fig_path_dict = umap_calc_and_save_html(
                        ct_df.set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                        ct_emb_columns, ct_save_dir, obs_columns, max_cells=200000, opacity=max((1 - ct_df.shape[0] / 200000), 0.3) # more opacity for fewer cells
                    )
                    # update fig_path_dict keys to include celltype expert index
                    fig_path_dict = {f"ct_{i}/{k}": v for k, v in fig_path_dict.items()}

                    for key, fig_path in fig_path_dict.items():
                        metrics_to_log[key] = wandb.Image(fig_path, caption=key)

                if self.save_validation_embeddings:
                    # remove old embeddings
                    if os.path.exists(os.path.join(self.root_dir, f"epoch_{self.current_epoch - 1}_validation_embeddings.csv")):
                        os.remove(os.path.join(self.root_dir, f"epoch_{self.current_epoch - 1}_validation_embeddings.csv"))
                    embeddings_df.to_csv(os.path.join(self.root_dir, f"epoch_{self.current_epoch}_validation_embeddings.csv"), index=False)

                if self.run_validation_metrics:
                    # Run metrics
                    metrics_dict = {}
                    metrics_keys = [
                        'acc_knn', 'kni', 'mean_pct_same_batch_in_knn', 'pct_cells_with_diverse_knn',
                        'acc_radius', 'rbni', 'mean_pct_same_batch_in_radius', 'pct_cells_with_diverse_radius'
                    ]

                    kni_results = calc_kni_score(
                        embeddings_df.set_index("soma_joinid")[emb_columns], 
                        self.obs_df.loc[embeddings_df.index], 
                        batch_col="study_name", 
                        n_neighbours=15, 
                        max_prop_same_batch=0.8, 
                        use_faiss=False
                    )
                    
                    rbni_results = calc_rbni_score(
                        embeddings_df.set_index("soma_joinid")[emb_columns], 
                        self.obs_df.loc[embeddings_df.index], 
                        batch_col="study_name", 
                        radius=1.0, 
                        max_prop_same_batch=0.8
                    )

                    for k, v in kni_results.items():
                        if k in metrics_keys:
                            metrics_dict[f"val_metrics/{k}"] = v

                    for k, v in rbni_results.items():
                        if k in metrics_keys:
                            metrics_dict[f"val_metrics/{k}"] = v

                    metrics_to_log.update(metrics_dict)
            
            self.valid_counter += 1
        
        # Clear validation outputs for next epoch
        self.validation_z_cell_refined = []
        self.validation_z_cell_list = []
        
        # Log metrics to wandb
        if metrics_to_log:
            wandb.log(metrics_to_log)

    def test_step(self, batch, batch_idx):
        """Lightning hook for test step"""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean=True):
        """Generate embeddings for inference"""
        outputs = self.forward(batch)
        
        # Return the refined modality embedding
        z = outputs["z_cell_refined"]
        
        # Important: make z float64 dtype to concat properly with soma_joinid
        z = z.double()
        
        if "soma_joinid" in batch:
            # Join z with soma_joinid
            return torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)
        elif "locate" in batch:
            return torch.cat((z, batch["locate"]), 1)
        else:
            return z

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""


        g_optimizer = torch.optim.Adam(self.model.g_params, **self.training_args["g_optimizer"])

        g_scheduler = ReduceLROnPlateau(
            g_optimizer,
            mode='min',           
            factor=0.5,           # halve the learning rate
            patience=4,           # wait 4 epochs for improvement
            min_lr=1e-6,
        )


  
        d_optimizer = torch.optim.Adam(self.model.d_params, **self.training_args["d_optimizer"])

        d_scheduler = ReduceLROnPlateau(
            d_optimizer,
            mode='min',           
            factor=0.5,           # halve the learning rate
            patience=4,           # wait 4 epochs for improvement
            min_lr=1e-6,
        )



        return [
            {
                "optimizer": g_optimizer,
                "lr_scheduler": {
                    "scheduler": g_scheduler,
                    "monitor": "val_loss/loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            },
            {
                "optimizer": d_optimizer,
                "lr_scheduler": {
                    "scheduler": d_scheduler,
                    "monitor": "val_loss/disc_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        ]
