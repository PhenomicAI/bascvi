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
from ml_benchmarking.bascvi.utils.utils import umap_calc_and_save_html, calc_kni_score, calc_rbni_score
from ml_benchmarking.mm_bascvi.model.distributions import ZeroInflatedNegativeBinomial


from ml_benchmarking.mm_bascvi.model.MMBAscVI import MMBAscVI  # Adjust import to your structure

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MMBAscVI_Trainer(pl.LightningModule):
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
        model_args: dict = {},
        training_args: dict = {},
        callbacks_args: dict = {},
    ):
        super().__init__()
        # Save hyperparameters in hparams.yaml file
        self.save_hyperparameters(ignore="datamodule")
        
        self.root_dir = root_dir
        self.valid_counter = 0
        
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
        
        # Create the model
        n_genes = n_genes or model_args.get("n_genes")
        n_latent = model_args.get("n_latent")
        n_modalities = model_args.get("n_modalities")
        num_cell_type_experts = model_args.get("num_cell_type_experts")
        
        self.model = MMBAscVI(
            n_genes=n_genes,
            latent_dim=n_latent,
            num_modalities=n_modalities,
            num_cell_type_experts=num_cell_type_experts
        )
        
        # Configure callbacks
        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize MMBAscVI Trainer")
        
        self.automatic_optimization = False  # PyTorch Lightning 2 method
        
        self.validation_step_outputs = []
        
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
        modality_vec = batch["modality_vec"]
        return self.model(x, modality_vec)
    
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
    
    def compute_reconstruction_loss(self, x_decoder_zinb_params, x, feature_presence_mask) -> torch.Tensor:
        """Compute reconstruction loss between x_reconstructed and x"""
        px_rate, px_r, px_dropout = x_decoder_zinb_params
        log_prob = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x)
        return log_prob * feature_presence_mask
    
    def compute_deconvolution_loss(self, z_modality_refined, cell_attn_weights, z_celltype_list) -> torch.Tensor:
        """
        Computes the deconvolution loss:
        L_deconv = || z_modality_refined - sum_j(cell_attn_weights * z_celltype_list) ||^2
        """
        # Stack the cell-type embeddings: [batch_size, num_ct, latent_dim]
        z_cell_stack = torch.stack(z_celltype_list, dim=1)  # [batch_size, num_ct, latent_dim]

        # Expand the attention weights to match the cell-type embedding dimensions
        alpha_expanded = cell_attn_weights.unsqueeze(-1)  # [batch_size, num_ct, 1]

        # Compute the weighted sum of cell-type embeddings
        celltype_combined = (alpha_expanded * z_cell_stack).sum(dim=1)  # [batch_size, latent_dim]

        # Compute the L2 difference between the refined modality embedding and the weighted sum
        l2_per_sample = (z_modality_refined - celltype_combined).pow(2).sum(dim=1)

        # Compute the mean loss across all samples
        deconv_loss = l2_per_sample.mean() 

        return deconv_loss

    def compute_vae_kl_loss(self, mod_vae_params, cell_vae_params):
        """Compute KL divergence loss for both modality and cell-type VAEs"""
        kl_loss = 0.0
        # Sum across all modality experts
        for (mu, logvar) in mod_vae_params:
            kl = 0.5 * torch.sum(logvar.exp() + mu**2 - 1.0 - logvar, dim=1).mean()
            kl_loss += kl

        # Sum across all cell-type experts
        for (mu, logvar) in cell_vae_params:
            kl = 0.5 * torch.sum(logvar.exp() + mu**2 - 1.0 - logvar, dim=1).mean()
            kl_loss += kl
            
        return kl_loss * self.kl_warmup_weight * self.kl_loss_weight

    def compute_modality_classification_loss(self, mod_logits_list, refined_mod_logits, modality_idx):
        """Compute modality classification loss from ModalityExperts"""
        ce_modality_raw = 0.0
        for logits_m in mod_logits_list:
            ce_modality_raw += F.cross_entropy(logits_m, modality_idx)

        ce_modality_refined = F.cross_entropy(refined_mod_logits, modality_idx)

        return 0.5 * ce_modality_raw + 0.5 * ce_modality_refined

    def compute_celltype_modality_loss(self, cell_logits_list, modality_idx):
        """Compute modality classification loss from CellTypeExperts"""
        modality_class_loss_celltype = 0.0
        for logits_c in cell_logits_list:
            ce = nn.functional.cross_entropy(logits_c, modality_idx)
            modality_class_loss_celltype += ce
        return modality_class_loss_celltype
    
    def compute_celltype_diversity_loss(self, z_celltype_list):
        """Celltype diversity loss: Minimize off-diagonal similarity; keep cell-type embeddings distinct"""
        # Convert list to tensor if it's a list
        if isinstance(z_celltype_list, list):
            z_celltype_tensor = torch.stack(z_celltype_list, dim=0)
        else:
            z_celltype_tensor = z_celltype_list
            
        # Normalize embeddings
        z_norm = F.normalize(z_celltype_tensor, p=2, dim=1)
        
        # Compute pairwise similarity between cell-type embeddings
        similarity_matrix = torch.matmul(z_norm, z_norm.transpose(0, 1))  # Cosine similarity

        # Create an identity matrix of the same size
        identity = torch.eye(z_norm.shape[0], device=z_norm.device)

        # Compute the mean off-diagonal similarity
        off_diagonal = similarity_matrix * (1 - identity)

        return off_diagonal.sum() / (z_norm.shape[0] * (z_norm.shape[0] - 1))  # Minimize off-diagonal similarity

    def compute_loss(self, outputs, batch):
        """Combined loss function for MMBAscVI model"""
        # 1) Unpack model outputs
        z_mod_list = outputs["z_modality_list"]
        mod_vae_params = outputs["modality_vae_params"]
        mod_logits_list = outputs["modality_logits_list"]
        refined_mod_logits = outputs["refined_modality_logits"]
        z_modality_refined = outputs["z_modality_refined"]
        cell_attn_weights = outputs["cell_attn_weights"]

        z_celltype_list = outputs["z_celltype_list"]
        cell_vae_params = outputs["celltype_vae_params"]
        cell_logits_list = outputs["celltype_logits_list"]
        
        # Extract batch data
        x = batch["x"]
        modality_idx = batch["modality_idx"]
        cell_type_labels = batch.get("cell_type_label", None)

        # Compute losses
        kl_loss = self.compute_vae_kl_loss(mod_vae_params, cell_vae_params)
        
        modality_class_loss = self.compute_modality_classification_loss(
            mod_logits_list, refined_mod_logits, modality_idx)
        
        modality_class_loss_celltype = self.compute_celltype_modality_loss(
            cell_logits_list, modality_idx)
        
        deconv_loss = self.compute_deconvolution_loss(
            z_modality_refined, cell_attn_weights, z_celltype_list)
        
        celltype_diversity_loss = self.compute_celltype_diversity_loss(z_celltype_list)

        # Combine losses with weights from training args
        mod_class_weight = self.training_args.get("modality_class_weight", 1.0)
        celltype_class_weight = self.training_args.get("celltype_class_weight", 1.0)
        deconv_weight = self.training_args.get("deconv_weight", 1.0)
        diversity_weight = self.training_args.get("diversity_weight", 1.0)
        
        total_loss = (
            kl_loss + 
            mod_class_weight * modality_class_loss - 
            celltype_class_weight * modality_class_loss_celltype + 
            deconv_weight * deconv_loss + 
            diversity_weight * celltype_diversity_loss
        )
        
        loss_dict = {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "modality_class_loss": modality_class_loss,
            "modality_class_loss_celltype": modality_class_loss_celltype,
            "deconv_loss": deconv_loss,
            "celltype_diversity_loss": celltype_diversity_loss
        }
        
        return loss_dict

    def training_step(self, batch, batch_idx):
        
        """Lightning hook that runs one training step"""
        # Skip small batches
        if batch["x"].shape[0] < 3:
            return None
        
        optimizer = self.optimizers()
        
        optimizer.zero_grad()
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs, batch)
        self.manual_backward(losses["loss"])
        
        # Clip gradients
        utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        # Log training losses
        train_losses = {f"train_loss/{k}": v for k, v in losses.items()}
        self.log_dict(train_losses)
        
        # Log trainer metrics
        self.log("trainer/kl_warmup_weight", self.kl_warmup_weight)
        self.log("trainer/global_step", self.global_step)
        
        return losses

    def validation_step(self, batch, batch_idx):
        """Lightning hook for validation step"""
        outputs = self.forward(
            x=batch["x"],
            modality_idx=batch["modality_idx"]
        )
        losses = self.compute_loss(outputs, batch)
        
        # Log validation losses
        val_losses = {f"val_loss/{k}": v for k, v in losses.items()}
        self.log_dict(val_losses, on_step=False, on_epoch=True)
        
        # For UMAP visualization, extract latent representations
        z_modality_refined = outputs["z_modality_refined"]
        
        # Make z float64 dtype to concat properly with soma_joinid
        z = z_modality_refined.double()
        
        if "soma_joinid" in batch:
            emb_output = torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)
            self.validation_step_outputs.append(emb_output)
        
        return losses

    def on_validation_epoch_end(self):
        """Process validation results at the end of the validation epoch"""
        metrics_to_log = {}
        
        if self.save_validation_umaps and self.validation_step_outputs:
            logger.info("Running validation UMAP...")
            embeddings = torch.cat(self.validation_step_outputs, dim=0).double().detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
           
            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            os.makedirs(save_dir, exist_ok=True)

            obs_columns = ["standard_true_celltype", "study_name", "sample_name", "scrnaseq_protocol"]
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
                
                _, fig_path_dict = umap_calc_and_save_html(
                    embeddings_df.set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns, save_dir, obs_columns, max_cells=100000
                )

                for key, fig_path in fig_path_dict.items():
                    metrics_to_log[key] = wandb.Image(fig_path, caption=key)

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
        self.validation_step_outputs.clear()
        
        # Log metrics to wandb
        if metrics_to_log:
            wandb.log(metrics_to_log)

    def test_step(self, batch, batch_idx):
        """Lightning hook for test step"""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean=True):
        """Generate embeddings for inference"""
        outputs = self.forward(
            x=batch["x"], 
            modality_idx=batch["modality_idx"]
        )
        
        # Return the refined modality embedding
        z = outputs["z_modality_refined"]
        
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
        optimizer = torch.optim.Adam(self.parameters(), **self.training_args.get("optimizer", {"lr": 1e-3}))
        config = {"optimizer": optimizer}
        
        if self.training_args.get("reduce_lr_on_plateau"):
            scheduler = ReduceLROnPlateau(
                optimizer,
                **self.training_args.get("reduce_lr_on_plateau"),
                threshold_mode="abs",
                verbose=True,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.training_args.get("lr_scheduler_metric", "val_loss/loss"),
            }
        
        elif self.training_args.get("step_lr_scheduler"):
            scheduler = StepLR(optimizer, **self.training_args.get("step_lr_scheduler"))
            config["lr_scheduler"] = scheduler
        
        return config
