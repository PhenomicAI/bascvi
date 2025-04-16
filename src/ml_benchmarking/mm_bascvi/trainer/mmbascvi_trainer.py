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
from ml_benchmarking.mm_bascvi.trainer.AsyncUMAPGenerator import AsyncUMAPGenerator
from ml_benchmarking.mm_bascvi.model.losses import MMBAscVILoss

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

        self.max_validation_batches_umap = 300 # TODO: change to take into account batch size
        
        # Create validation_umaps directory if needed
        if os.path.isdir(os.path.join(self.root_dir, "validation_umaps")):
            shutil.rmtree(os.path.join(self.root_dir, "validation_umaps"))

        self.async_umap = AsyncUMAPGenerator()

        # Initialize loss calculator
        self.loss_calculator = MMBAscVILoss(
            loss_weights=training_args.get("loss_weights", {}),
            bulk_id=self.bulk_id,
            training_args=training_args
        )

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
        bulk_mask = batch["batch_idx"][:, 0] == self.bulk_id
        return self.model(x, batch_idx, bulk_mask=bulk_mask)
    

    def training_step(self, batch, batch_idx):
        """Lightning hook that runs one training step"""

        in_warmup_cache = self.loss_calculator.is_in_warmup(self.global_step)

        # Skip small batches
        if batch["x"].shape[0] < 3:
            return None
        
        self.log("trainer/global_step", self.global_step)

        g_opt, d_opt = self.optimizers()
        g_scheduler, d_scheduler = self.lr_schedulers()

        g_opt.zero_grad()
        g_outputs = self.forward(batch)
        g_loss, g_loss_components = self.loss_calculator.compute_loss(g_outputs, batch, optimizer_idx=0, global_step=self.global_step)
        self.manual_backward(g_loss)
        utils.clip_grad_norm_(self.model.parameters(), 1.0) # clip gradients
        g_opt.step()

        # Log training losses
        g_loss_components["loss"] = g_loss
        train_losses = {f"train/g/{k}": v for k, v in g_loss_components.items()}
        self.log_dict(train_losses)

        self.log("trainer/vae_lr", g_opt.param_groups[0]['lr'])

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

        if self.training_args['loss_weights']['adversarial'] > 0.0:
            d_opt.zero_grad()
            d_outputs = self.forward(batch)
            d_loss, d_loss_components = self.loss_calculator.compute_loss(d_outputs, batch, optimizer_idx=1, global_step=self.global_step)
            self.manual_backward(d_loss)
            utils.clip_grad_norm_(self.model.parameters(), 1.0)
            d_opt.step()

            # Log discriminator losses
            d_loss_components["loss"] = d_loss
            d_losses = {f"train/d/{k}": v for k, v in d_loss_components.items()}
            self.log_dict(d_losses)

            self.log("trainer/discriminator_lr", d_opt.param_groups[0]['lr'])

            if not in_warmup_cache:
                for i in range(len(self.batch_dict_keys)):
                    loss_percentage[f"loss_pct/disc_ct_{self.batch_dict_keys[i]}"] = self.training_args["loss_weights"]["ct_discriminator"][i] * g_loss_components[f"disc_ct_{self.batch_dict_keys[i]}"].item() / g_loss.item()
                    loss_percentage[f"loss_pct/disc_z_{self.batch_dict_keys[i]}"] = self.training_args["loss_weights"]["z_discriminator"][i] * g_loss_components[f"disc_z_{self.batch_dict_keys[i]}"].item() / g_loss.item()


        self.log_dict(loss_percentage)
            
        return g_loss

    def on_train_epoch_end(self):
        """Called at the end of the training epoch"""
        # Lightning will handle scheduler stepping with the configuration above
        # No need to manually step schedulers here
        pass

    def validation_step(self, batch, batch_idx):
        """Lightning hook for validation step"""
        outputs = self.forward(batch)
        loss, loss_components = self.loss_calculator.compute_loss(outputs, batch, optimizer_idx=0, global_step=self.global_step)
        disc_loss, disc_loss_components = self.loss_calculator.compute_loss(outputs, batch, optimizer_idx=1, global_step=self.global_step)

        # Log encoder losses
        loss_components["loss"] = loss
        val_losses = {f"val/g/{k}": v for k, v in loss_components.items()}

        # Log discriminator losses
        disc_loss_components["loss"] = disc_loss
        val_losses.update({f"val/d/{k}": v for k, v in disc_loss_components.items()})

        self.log_dict(val_losses, on_step=False, on_epoch=True)
        
        if self.save_validation_umaps and ((len(self.validation_z_cell_refined) == 0) or (len(self.validation_z_cell_refined) < self.max_validation_batches_umap)):
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

        # Just check status of pending UMAP tasks
        pending_count = self.async_umap.check_pending()
        if pending_count > 0:
            logger.info(f"Currently {pending_count} UMAP generation tasks pending")

        if self.save_validation_umaps and self.validation_z_cell_refined:
            logger.info("Submitting validation UMAP tasks asynchronously...")
            embeddings = torch.cat(self.validation_z_cell_refined, dim=0).double().detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])

            soma_joinid = embeddings_df["soma_joinid"].values

            # Sample rows for UMAP
            if embeddings_df.shape[0] > self.training_args.get("downsample_umaps_to_n_cells", 100000):
                downsampled_idx = np.random.choice(embeddings_df.shape[0], size=self.training_args.get("downsample_umaps_to_n_cells", 100000), replace=False)
            else:
                downsampled_idx = np.arange(embeddings_df.shape[0])

            # Process cell type embeddings
            ct_embeddings_list = []
            for batch_embeddings in self.validation_z_cell_list:
                stacked_batch = torch.stack(batch_embeddings, dim=1)
                ct_embeddings_list.append(stacked_batch)
            
            ct_embeddings = torch.cat(ct_embeddings_list, dim=0).double().detach().cpu().numpy()
            num_celltype_experts = ct_embeddings.shape[1]
            num_ct_latent_dims = ct_embeddings.shape[2]

            # Modality embeddings
            modality_embeddings = torch.cat(self.validation_z_modality_refined, dim=0).double().detach().cpu().numpy()
            modality_df = pd.DataFrame(data=modality_embeddings, columns=emb_columns)
            modality_df["soma_joinid"] = soma_joinid

            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            obs_columns = ["standard_true_celltype", "study_name", "sample_name", "scrnaseq_protocol", "tissue_collected"]

            if self.obs_df is None and self.soma_experiment_uri:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    self.obs_df = soma_experiment.obs.read(column_names=['soma_joinid'] + obs_columns).concat().to_pandas()
                    self.obs_df = self.obs_df.set_index("soma_joinid")
            
            if self.obs_df is not None:
                # Submit main embeddings UMAP task - now with direct wandb logging
                self.async_umap.submit_umap_task(
                    key_prefix="z",
                    embeddings_df=embeddings_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns=emb_columns,
                    save_dir=save_dir,
                    obs_df=self.obs_df,
                    obs_columns=obs_columns,
                    epoch=self.current_epoch,
                    step=self.global_step,
                    max_cells=100000,
                    opacity=max((1 - embeddings_df.iloc[downsampled_idx].shape[0] / 100000), 0.2)
                )

                # Submit modality embeddings UMAP task
                modality_save_dir = os.path.join(save_dir, "modality_umaps")
                self.async_umap.submit_umap_task(
                    key_prefix="modality",
                    embeddings_df=modality_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                    emb_columns=emb_columns,
                    save_dir=modality_save_dir,
                    obs_df=self.obs_df,
                    obs_columns=obs_columns,
                    epoch=self.current_epoch,
                    step=self.global_step,
                    max_cells=100000,
                    opacity=max((1 - modality_df.iloc[downsampled_idx].shape[0] / 100000), 0.2)
                )

                # Submit celltype embeddings UMAP tasks
                for i in range(min(num_celltype_experts, 10)):
                    ct_save_dir = os.path.join(save_dir, "celltype_umaps", f"ct_{i}")
                    ct_emb_columns = [f"ct_{i}_dim_{j}" for j in range(num_ct_latent_dims)]
                    ct_df = pd.DataFrame(data=ct_embeddings[:, i, :], columns=ct_emb_columns)
                    ct_df["soma_joinid"] = soma_joinid

                    self.async_umap.submit_umap_task(
                        key_prefix=f"ct_{i}",
                        embeddings_df=ct_df.iloc[downsampled_idx].set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(),
                        emb_columns=ct_emb_columns,
                        save_dir=ct_save_dir,
                        obs_df=self.obs_df,
                        obs_columns=obs_columns,
                        epoch=self.current_epoch,
                        step=self.global_step,
                        max_cells=100000,
                        opacity=max((1 - ct_df.iloc[downsampled_idx].shape[0] / 100000), 0.2)
                    )

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
        
        # Log any metrics calculated synchronously
        if metrics_to_log:
            wandb.log(metrics_to_log, step=self.global_step)

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
            threshold=1.0
        )
        
        d_optimizer = torch.optim.Adam(self.model.d_params, **self.training_args["d_optimizer"])
        d_scheduler = ReduceLROnPlateau(
            d_optimizer,
            mode='min',           
            factor=0.5,           # halve the learning rate
            patience=4,           # wait 4 epochs for improvement
            min_lr=1e-6,
            threshold=1.0
        )

        return [
            {
                "optimizer": g_optimizer,
                "lr_scheduler": {
                    "scheduler": g_scheduler,
                    "monitor": "val/g/loss",  # This tells Lightning which metric to monitor
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,  # Set to False to avoid errors if val/g/loss isn't available
                },
            },
            {
                "optimizer": d_optimizer,
                "lr_scheduler": {
                    "scheduler": d_scheduler,
                    "monitor": "val/d/loss",  # Monitor the discriminator loss for its scheduler
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                },
            },
        ]
