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
from ml_benchmarking.bascvi.utils.utils import umap_calc_and_plot, calc_kni_score, calc_rbni_score
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

        self.loss_weights = training_args.get("loss_weights", {})
        
        # Get specific training parameters
        self.n_steps_kl_ramp = training_args.get("n_steps_kl_ramp", 0.0)
        self.n_steps_adv_ramp = training_args.get("n_steps_adv_ramp", 0.0)
        self.n_steps_adv_start = training_args.get("n_steps_adv_start", 0.0)
        
        self.save_validation_umaps = training_args.get("save_validation_umaps", False)
        self.run_validation_metrics = training_args.get("run_validation_metrics", False)
        self.save_validation_embeddings = training_args.get("save_validation_embeddings", False)

        self.modalities_idx_to_name_dict = modalities_idx_to_name_dict

        # search for "Bulk" in modalities_idx_to_name_dict values, get key
        self.bulk_id = next((key for key, value in self.modalities_idx_to_name_dict.items() if "Bulk" in value), None)

        self.model_args["bulk_id"] = self.bulk_id
        self.model = MMBAscVI(**self.model_args)
        
        # Configure callbacks
        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize MMBAscVI Trainer")
        
        self.automatic_optimization = False  # PyTorch Lightning 2 method
        
        self.validation_z_cell_refined = []
        self.validation_z_cell_list = []
        self.validation_z_modality_refined = []

        self.batch_dict_keys = ['modality', 'study', 'sample']
        
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
    

    def is_in_warmup(self):
        """Helper to check if we're in warm-up phase"""
        return self.global_step < self.n_steps_adv_start
    

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

    def get_loss_weights(self):
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
        if self.n_steps_kl_ramp > 0 and self.global_step < self.n_steps_kl_ramp:
            ramp_factor = self.global_step / self.n_steps_kl_ramp
            weights["kl"] *= ramp_factor
        
        # Apply adversarial weight adjustment
        if self.global_step < self.n_steps_adv_start:
            weights["adversarial"] = 0.0
        elif self.global_step < self.n_steps_adv_start + self.n_steps_adv_ramp:
            ramp_factor = (self.global_step - self.n_steps_adv_start) / self.n_steps_adv_ramp
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

            # weight the bulk loss by 0
            log_prob_bulk = log_prob_bulk * self.training_args["loss_weights"]["bulk_reconstruction"]

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
        #cell_kl_loss = cell_kl_loss / len(cell_vae_params)

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
        # We want to minimize similarity, not maximize it
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
        desired_rank = self.training_args.get("desired_ct_rank", 2)
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
    
    def compute_loss(self, outputs, batch, optimizer_idx=0):
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
        loss_weights = self.get_loss_weights()

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
            if not self.is_in_warmup():
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
                    total_loss -= loss_weights["adversarial"] * loss_weights["ct_discriminator"][i] * loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"] 
                    total_loss -= loss_weights["adversarial"] * loss_weights["z_discriminator"][i] * loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"] 

            
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
                total_loss += loss_weights["ct_discriminator"][i] * loss_components_dict[f"disc_ct_{self.batch_dict_keys[i]}"]
                total_loss += loss_weights["z_discriminator"][i] * loss_components_dict[f"disc_z_{self.batch_dict_keys[i]}"]

        return total_loss, loss_components_dict

    def training_step(self, batch, batch_idx):
        """Lightning hook that runs one training step"""

        in_warmup_cache = self.is_in_warmup()

        # Skip small batches
        if batch["x"].shape[0] < 3:
            return None

        
        g_opt, d_opt = self.optimizers()
        g_scheduler, d_scheduler = self.lr_schedulers()

        g_opt.zero_grad()
        g_outputs = self.forward(batch)
        g_loss, g_loss_components = self.compute_loss(g_outputs, batch, optimizer_idx=0)
        self.manual_backward(g_loss)
        utils.clip_grad_norm_(self.model.parameters(), 1.0) # clip gradients
        g_opt.step()

        d_opt.zero_grad()
        d_outputs = self.forward(batch)
        d_loss, d_loss_components = self.compute_loss(d_outputs, batch, optimizer_idx=1)
        self.manual_backward(d_loss)
        # clip gradients
        utils.clip_grad_norm_(self.model.parameters(), 1.0) # clip gradients
        d_opt.step()

        # Log training losses
        g_loss_components["loss"] = g_loss
        train_losses = {f"train/g/{k}": v for k, v in g_loss_components.items()}
        self.log_dict(train_losses)

        # Log discriminator losses
        d_loss_components["loss"] = d_loss
        d_losses = {f"train/d/{k}": v for k, v in d_loss_components.items()}
        self.log_dict(d_losses)
        
        # Log trainer metrics
        # self.log("trainer/kl_warmup_weight", self.kl_warmup_weight)
        self.log("trainer/global_step", self.global_step)

        # Get current learning rates        
        self.log("trainer/vae_lr", g_opt.param_groups[0]['lr'])
        self.log("trainer/discriminator_lr", d_opt.param_groups[0]['lr'])

        # Log attention temps
        log_temps_per_modality = self.model.celltype_cross_attention.log_temps
        for i, temp in enumerate(log_temps_per_modality):
            self.log(f"trainer/ct_attn_temp_{self.modalities_idx_to_name_dict[i]}", torch.exp(temp))
        
        
        # Log loss percentages
        loss_percentage = {
            "loss_pct/reconstruction": self.training_args["loss_weights"]["reconstruction"] * g_loss_components["reconstruction"].item() / g_loss.item(),
            "loss_pct/kl": self.training_args["loss_weights"]["kl"] * g_loss_components["kl"].item() / g_loss.item(),
            "loss_pct/ct_diversity": self.training_args["loss_weights"]["ct_diversity"] * g_loss_components["ct_diversity"].item() / g_loss.item(),
            "loss_pct/ct_regularization": self.training_args["loss_weights"]["ct_regularization"] * g_loss_components["ct_regularization"].item() / g_loss.item(),
        }

        if not in_warmup_cache:
            for i in range(len(self.batch_dict_keys)):
                loss_percentage[f"loss_pct/disc_ct_{self.batch_dict_keys[i]}"] = self.training_args["loss_weights"]["ct_discriminator"][i] * g_loss_components[f"disc_ct_{self.batch_dict_keys[i]}"].item() / g_loss.item()
                loss_percentage[f"loss_pct/disc_z_{self.batch_dict_keys[i]}"] = self.training_args["loss_weights"]["z_discriminator"][i] * g_loss_components[f"disc_z_{self.batch_dict_keys[i]}"].item() / g_loss.item()

        
        self.log_dict(loss_percentage)
            
        return g_loss

    def on_train_epoch_end(self):
        """Called at the end of the training epoch"""
        # Get the current validation loss to update the schedulers
        g_scheduler, d_scheduler = self.lr_schedulers()
        
        # Step the schedulers based on the validation loss
        # Note: You may want to save these metrics from your validation step
        if hasattr(self, 'val_loss'):
            g_scheduler.step(self.val_loss)
            d_scheduler.step(self.val_disc_loss if hasattr(self, 'val_disc_loss') else self.val_loss)

    def validation_step(self, batch, batch_idx):
        """Lightning hook for validation step"""
        outputs = self.forward(batch)
        loss, loss_components = self.compute_loss(outputs, batch)
        disc_loss, disc_loss_components = self.compute_loss(outputs, batch, optimizer_idx=1)

        # Log encoder losses
        loss_components["loss"] = loss
        val_losses = {f"val/g/{k}": v for k, v in loss_components.items()}

        # Log discriminator losses
        disc_loss_components["loss"] = disc_loss
        val_losses.update({f"val/d/{k}": v for k, v in disc_loss_components.items()})

        self.log_dict(val_losses, on_step=False, on_epoch=True)
        
        # For UMAP visualization, extract latent representations
        z_cell_refined = outputs["z_cell_refined"]
        z_cell_list = outputs["z_celltype_list"]
        z_modality_refined = outputs["z_modality_refined"]
        
        # Make z float64 dtype to concat properly with soma_joinid
        z = z_cell_refined.double()
        
        if "soma_joinid" in batch:
            emb_output = torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)
            self.validation_z_cell_refined.append(emb_output)
            self.validation_z_cell_list.append(z_cell_list)
            self.validation_z_modality_refined.append(z_modality_refined)
        return loss

    def on_validation_epoch_end(self):
        """Process validation results at the end of the validation epoch"""
        # Store validation loss for scheduler
        if self.trainer.callback_metrics.get('g/val/loss') is not None:
            self.val_loss = self.trainer.callback_metrics['g/val/loss']
        if self.trainer.callback_metrics.get('d/val/loss') is not None:
            self.val_disc_loss = self.trainer.callback_metrics['d/val/loss']
        
        metrics_to_log = {}

        if self.save_validation_umaps and self.validation_z_cell_refined:
            logger.info("Running validation UMAP...")
            embeddings = torch.cat(self.validation_z_cell_refined, dim=0).double().detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

            soma_joinid = embeddings_df["soma_joinid"].values

            # sample rows for UMAP
            if embeddings_df.shape[0] > self.training_args.get("downsample_umaps_to_n_cells", 100000):
                downsampled_idx = np.random.choice(embeddings_df.shape[0], size=self.training_args.get("downsample_umaps_to_n_cells", 100000), replace=False)
            else:
                downsampled_idx = np.arange(embeddings_df.shape[0])

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
            num_ct_latent_dims = ct_embeddings.shape[2]


            # Modality UMAPs
            modality_embeddings = torch.cat(self.validation_z_modality_refined, dim=0).double().detach().cpu().numpy()
            modality_df = pd.DataFrame(data=modality_embeddings, columns=emb_columns)
            modality_df["soma_joinid"] = soma_joinid

           
            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            os.makedirs(save_dir, exist_ok=True)


            obs_columns = ["standard_true_celltype", "study_name", "sample_name", "scrnaseq_protocol", "tissue_collected"]
            # if "species" not in obs_columns:
            #     obs_columns.append("species")

            if self.obs_df is None and self.soma_experiment_uri:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    self.obs_df = soma_experiment.obs.read(column_names=['soma_joinid'] + obs_columns).concat().to_pandas()
                    self.obs_df = self.obs_df.set_index("soma_joinid")
            
            if self.obs_df is not None:
                # if "species" not in self.obs_df.columns:
                #     logger.info("Adding species column to obs, assuming human data")
                #     self.obs_df["species"] = "human"

                #     # assign species based on study_name
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_m-"), "species"] = "mouse"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_r-"), "species"] = "rat"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_l-"), "species"] = "lemur"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_c-"), "species"] = "macaque"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_f-"), "species"] = "fly"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_a-"), "species"] = "axolotl"
                #     self.obs_df.loc[self.obs_df["study_name"].str.contains("_z-"), "species"] = "zebrafish"
                


                _, _, fig_path_dict = umap_calc_and_plot(
                    embeddings_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns, save_dir, obs_columns, max_cells=200000, opacity=max((1 - embeddings_df.iloc[downsampled_idx].shape[0] / 100000), 0.3) # more opacity for fewer cells
                ) 
                for key, fig_path in fig_path_dict.items():
                    metrics_to_log[f"z/{key}"] = wandb.Image(fig_path)

                modality_save_dir = os.path.join(save_dir, "modality_umaps")
                os.makedirs(modality_save_dir, exist_ok=True)

                # Modality UMAPs
                _, _, fig_path_dict = umap_calc_and_plot(
                    modality_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns, modality_save_dir, obs_columns, max_cells=200000, opacity=max((1 - modality_df.iloc[downsampled_idx].shape[0] / 100000), 0.3) # more opacity for fewer cells
                )
                for key, fig_path in fig_path_dict.items():
                    metrics_to_log[f"modality/{key}"] = wandb.Image(fig_path)


                # Celltype UMAPs
                for i in range(min(num_celltype_experts, 10)): # only plot first 10 celltype experts
                    ct_save_dir = os.path.join(save_dir, "celltype_umaps", f"ct_{i}")
                    os.makedirs(ct_save_dir, exist_ok=True)
                    ct_emb_columns = [f"ct_{i}_dim_{j}" for j in range(num_ct_latent_dims)]
                    ct_df = pd.DataFrame(data=ct_embeddings[:, i, :], columns=ct_emb_columns)
                    
                    ct_df["soma_joinid"] = soma_joinid

                    _, _, fig_path_dict = umap_calc_and_plot(
                        ct_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                        ct_emb_columns, ct_save_dir, obs_columns, max_cells=200000, opacity=max((1 - ct_df.iloc[downsampled_idx].shape[0] / 100000), 0.3) # more opacity for fewer cells
                    )
                    # update fig_dict keys to include celltype expert index
                    fig_path_dict = {f"ct_{i}/{k}": v for k, v in fig_path_dict.items()}
                    for key, fig_path in fig_path_dict.items():
                        # log as Image 
                        metrics_to_log[key] = wandb.Image(fig_path)

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
        self.validation_z_modality_refined = []
        
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
            patience=8,           # wait 8 epochs for improvement
            min_lr=1e-6,
        )

        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]
