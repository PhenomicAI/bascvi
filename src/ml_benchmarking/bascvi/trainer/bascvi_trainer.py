import os
import shutil
import logging
import itertools
import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_kld_cycle(epoch, period=20):
    ct = epoch % period
    pt = epoch % (period//2)
    if ct >= period//2:
        return 1
    else:
        return min(1, (pt) / (period//2))    

class BAScVITrainer(pl.LightningModule):
    """Lightning module to train scvi-vae model."""
    def __init__(
        self,
        root_dir,
        soma_experiment_uri: str = None,
        model_args: dict = {},
        training_args: dict = {},
        callbacks_args: dict = {},
        module_name: str = "bascvi",
        class_name: str = "BAScVI",
        gene_list: list = None,
        n_input: int = None,
        batch_level_sizes: list = None,
        predict_only: bool = False,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="datamodule")
        self.root_dir = root_dir
        self.model_args = model_args
        if n_input is not None:
            self.model_args["n_input"] = n_input
        if batch_level_sizes is not None:
            self.model_args["batch_level_sizes"] = batch_level_sizes
        self.training_args = training_args
        self.callbacks_args = callbacks_args
        self.kl_loss_weight = training_args.get("kl_loss_weight")
        self.n_epochs_kl_warmup = training_args.get("n_epochs_kl_warmup")
        self.n_steps_kl_warmup = training_args.get("n_steps_kl_warmup")
        self.disc_loss_weight = training_args.get("disc_loss_weight")
        self.n_epochs_discriminator_warmup = training_args.get("n_epochs_discriminator_warmup")
        self.n_steps_discriminator_warmup = training_args.get("n_steps_discriminator_warmup")
        self.use_library = training_args.get("use_library", False)
        self.save_validation_umaps = training_args.get("save_validation_umaps", False)
        self.run_validation_metrics = training_args.get("run_validation_metrics", False)
        self.gene_list = gene_list
        self.predict_only = predict_only
        if predict_only:
            self.model_args["predict_only"] = predict_only
        module = __import__("ml_benchmarking.bascvi.model", globals(), locals(), [module_name], 0)
        Vae = getattr(module, class_name)
        self.vae = Vae(**model_args)
        logger.info(f"{module_name} model args:\n {model_args}")
        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize {module_name} Trainer.")
        self.automatic_optimization = False
        self.validation_step_outputs = []
        if os.path.isdir(os.path.join(self.root_dir, "validation_umaps")):
            shutil.rmtree(os.path.join(self.root_dir, "validation_umaps"))

        self.soma_experiment_uri = soma_experiment_uri
        if soma_experiment_uri:
            from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
            self.open_soma_experiment = open_soma_experiment


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        predict_only = kwargs.get('predict_only', False)
        model = super(BAScVITrainer, cls).load_from_checkpoint(
            checkpoint_path, 
            map_location=map_location, 
            strict=False,
            **kwargs
        )
        if predict_only:
            model.eval()
        return model

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.callbacks_args["model_checkpoint"]["monitor"],
            mode=self.callbacks_args["model_checkpoint"]["mode"],
            filename="scvi-vae-{epoch:02d}-{elbo_val:.2f}"
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

    def forward(self, batch, **kwargs):
        return self.vae(batch, **kwargs)

    @property
    def kl_warmup_weight(self):
        epoch_criterion = self.n_epochs_kl_warmup is not None
        step_criterion = self.n_steps_kl_warmup is not None
        cyclic_criterion = self.training_args.get("cyclic_kl_period", False)
        if epoch_criterion:
            return min(1.0, self.current_epoch / self.n_epochs_kl_warmup)
        elif step_criterion:
            return min(1.0, self.global_step / self.n_steps_kl_warmup)
        elif cyclic_criterion:
            return get_kld_cycle(self.current_epoch, period=cyclic_criterion)
        return 1.0
    
    @property
    def disc_warmup_weight(self):
        def sigmoid_warmup(epoch, max_epochs):
            beta = 10
            T = 1
            progress = epoch / max_epochs
            return 1 / (1 + np.exp(-beta * (progress - T / 2)))
        epoch_criterion = self.n_epochs_discriminator_warmup is not None
        step_criterion = self.n_steps_discriminator_warmup is not None
        sigmoidal_criterion = self.training_args.get("sigmoidal_disc_warmup")
        if epoch_criterion:
            if sigmoidal_criterion:
                return min(1.0, sigmoid_warmup(self.current_epoch, self.n_epochs_discriminator_warmup))
            return min(1.0, self.current_epoch / self.n_epochs_discriminator_warmup)
        elif step_criterion:
            if sigmoidal_criterion:
                return min(1.0, sigmoid_warmup(self.global_step, self.n_steps_discriminator_warmup))
            return min(1.0, self.global_step / self.n_steps_discriminator_warmup)
        return 1.0
        
    def training_step(self, batch, batch_idx):
        
        if batch["x"].shape[0] < 3:
            return None

        if self.training_args.get("train_adversarial"):

            g_opt, d_opt = self.optimizers()
            
            # Generator update
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            # Clip gradients only for generator parameters (encoder, decoder, not discriminators)
            generator_params = []
            generator_params.extend(self.vae.z_encoder.parameters())
            generator_params.extend(self.vae.decoder.parameters())
            if self.vae.use_library and hasattr(self.vae, 'l_encoder'):
                generator_params.extend(self.vae.l_encoder.parameters())
            generator_params.append(self.vae.px_r)  # Include px_r parameter
            # Use configurable gradient clipping max norm, default to 5.0 if not specified
            max_grad_norm = self.training_args.get("max_grad_norm", 5.0)
            utils.clip_grad_norm_(generator_params, max_grad_norm)
            g_opt.step()
            
            # Discriminator update - only train discriminator after warmup period
            # Skip discriminator training during warmup to stabilize early training
            # Use threshold of 0.01 to determine if warmup is substantial enough
            if self.disc_warmup_weight >= 0.01:
                d_opt.zero_grad()
                _, _, d_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=1)
                self.manual_backward(d_losses['loss'])
                # Clip gradients only for discriminator parameters
                discriminator_params = []
                if self.vae.z_predictors is not None:
                    for predictor in self.vae.z_predictors:
                        discriminator_params.extend(predictor.parameters())
                if self.vae.x_predictors is not None:
                    for predictor in self.vae.x_predictors:
                        discriminator_params.extend(predictor.parameters())
                if discriminator_params:
                    # Use configurable gradient clipping max norm, default to 5.0 if not specified
                    max_grad_norm = self.training_args.get("max_grad_norm", 5.0)
                    utils.clip_grad_norm_(discriminator_params, max_grad_norm)
                d_opt.step()

        else:
            g_opt = self.optimizers()
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            # Use configurable gradient clipping max norm, default to 5.0 if not specified
            max_grad_norm = self.training_args.get("max_grad_norm", 5.0)
            utils.clip_grad_norm_(self.vae.parameters(), max_grad_norm)
            g_opt.step()
        
        # Filter out individual discriminator losses to avoid serialization
        filtered_g_losses = {}
        for k, v in g_losses.items():
            # Skip lindividual discriminator losses
            if (k.startswith("disc_loss_z_") or k.startswith("disc_loss_x_")):
                continue
            elif k == "disc_loss":  # Keep the aggregated discriminator loss
                filtered_g_losses[f"train_loss/{k}"] = v
            else:
                filtered_g_losses[f"train_loss/{k}"] = v
        
        self.log_dict(filtered_g_losses)
        self.log("trainer/disc_warmup_weight", self.disc_warmup_weight)
        self.log("trainer/kl_warmup_weight", self.kl_warmup_weight)
        self.log("trainer/global_step", self.global_step)

        # Log training progress periodically
        if batch_idx % 100 == 0:  # Log every 100 batches
            self._log_training_progress(batch_idx, g_losses)

        return g_losses

    def on_validation_epoch_start(self):
        """Reset validation batch counter at the start of validation epoch."""
        self.validation_batch_count = 0

    def validation_step(self, batch):
        # Use full weights (no warmup) for validation to get accurate model performance
        # For validation loss, only include reconstruction loss and KL divergence (exclude discriminator loss)
        inference_outputs, _, g_losses = self.forward(batch, kl_warmup_weight=1.0, disc_loss_weight=0.0, disc_warmup_weight=1.0, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
        
        # Filter out large objects and individual discriminator losses to avoid serialization
        filtered_g_losses = {}
        for k, v in g_losses.items():
            # Skip large objects and individual discriminator losses
            if (k.startswith("disc_loss_z_") or k.startswith("disc_loss_x_")):
                continue
            elif k == "disc_loss":  # Keep the aggregated discriminator loss
                filtered_g_losses[f"val_loss/{k}"] = v
            else:
                filtered_g_losses[f"val_loss/{k}"] = v
        
        self.log_dict(filtered_g_losses, on_step=False, on_epoch=True)
        qz_m = inference_outputs["qz_m"]
        z = qz_m.double()
        emb_output = torch.cat((z, torch.unsqueeze(batch["cell_idx"], 1)), 1)
        self.validation_step_outputs.append(emb_output)

        # Log validation progress periodically
        if hasattr(self, 'validation_batch_count'):
            self.validation_batch_count += 1
        else:
            self.validation_batch_count = 1
        
        if self.validation_batch_count % 50 == 0:  # Log every 50 validation batches
            self._log_validation_progress(self.validation_batch_count, g_losses)

        return g_losses

    def on_validation_epoch_end(self):
        backend = getattr(self.datamodule, "backend", "soma")
        if backend == "soma" and self.soma_experiment_uri:
            with self.open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                pass  # (keep your current logic here)
        elif backend == "zarr":
            logger.info("Skipping SOMA validation: running with Zarr backend.")
            # Output validation statistics to stdout
            self._log_validation_stats()
        else:
            logger.warning("Unknown backend or missing SOMA experiment URI, skipping validation/metrics.")

    def _log_validation_stats(self):
        """Log validation statistics to stdout."""
        # Get the logged metrics from the current epoch
        logged_metrics = self.trainer.logged_metrics
        
        if not logged_metrics:
            logger.info("No validation metrics available for logging.")
            return
        
        # Filter validation metrics
        val_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('val_loss/')}
        
        if not val_metrics:
            logger.info("No validation loss metrics found.")
            return
        
        # Print validation statistics
        logger.info("=" * 60)
        logger.info(f"VALIDATION STATISTICS - Epoch {self.current_epoch}")
        logger.info("=" * 60)
        
        # Print key metrics first
        # Note: Total loss = rec_loss + kl_loss - (disc_loss_weight * disc_warmup_weight * disc_loss)
        # Discriminator loss is subtracted in adversarial training, which can make total loss negative
        for metric_name in ['val_loss/loss', 'val_loss/rec_loss', 'val_loss/kl_loss', 'val_loss/disc_loss']:
            if metric_name in val_metrics:
                metric_value = val_metrics[metric_name]
                if hasattr(metric_value, 'item'):
                    metric_value = metric_value.item()
                logger.info(f"{metric_name}: {metric_value:.6f}")
        
        # Print weighted discriminator loss (what's actually subtracted from total loss)
        # Note: Validation uses warmup_weight=1.0, so weighted_disc_loss = disc_weight * 1.0 * disc_loss
        if 'val_loss/disc_loss' in val_metrics:
            disc_loss = val_metrics['val_loss/disc_loss']
            if hasattr(disc_loss, 'item'):
                disc_loss = disc_loss.item()
            # Validation uses full weights (warmup=1.0)
            weighted_disc_loss = self.disc_loss_weight * 1.0 * disc_loss
            logger.info(f"val_loss/weighted_disc_loss (disc_weight * 1.0 * disc_loss): {weighted_disc_loss:.6f}")
        
        # Print other validation metrics
        key_metrics_set = {'val_loss/loss', 'val_loss/rec_loss', 'val_loss/kl_loss', 'val_loss/disc_loss'}
        other_val_metrics = {k: v for k, v in val_metrics.items() if k not in key_metrics_set}
        if other_val_metrics:
            logger.info("-" * 40)
            logger.info("OTHER VALIDATION METRICS:")
            for metric_name, metric_value in other_val_metrics.items():
                if hasattr(metric_value, 'item'):
                    metric_value = metric_value.item()
                logger.info(f"{metric_name}: {metric_value:.6f}")
        
        # Print training metrics for comparison
        train_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('train_loss/')}
        if train_metrics:
            logger.info("-" * 40)
            logger.info("TRAINING STATISTICS (for comparison):")
            for metric_name, metric_value in train_metrics.items():
                if hasattr(metric_value, 'item'):
                    metric_value = metric_value.item()
                logger.info(f"{metric_name}: {metric_value:.6f}")
        
        # Print trainer state information
        logger.info("-" * 40)
        logger.info("TRAINER STATE:")
        logger.info(f"Global Step: {self.global_step}")
        logger.info(f"Current Epoch: {self.current_epoch}")
        logger.info(f"Training KL Warmup Weight: {self.kl_warmup_weight:.4f}")
        logger.info(f"Training Disc Warmup Weight: {self.disc_warmup_weight:.4f}")
        logger.info(f"Validation uses full weights (warmup=1.0) for accurate performance assessment")
        logger.info(f"Disc Loss Weight: {self.disc_loss_weight:.4f}")
        logger.info(f"Disc Weighted Product (disc_weight * disc_warmup): {self.disc_loss_weight * self.disc_warmup_weight:.4f}")
        
        logger.info("=" * 60)

    def _log_training_progress(self, batch_idx, losses):
        """Log training progress to stdout."""
        # Get the main loss value
        main_loss = losses.get('loss', 0)
        if hasattr(main_loss, 'item'):
            main_loss = main_loss.item()
        
        # Get other important metrics
        kl_loss = losses.get('kl_loss', 0)
        if hasattr(kl_loss, 'item'):
            kl_loss = kl_loss.item()
        
        rec_loss = losses.get('rec_loss', 0)
        if hasattr(rec_loss, 'item'):
            rec_loss = rec_loss.item()
        
        disc_loss = losses.get('disc_loss', 0)
        if hasattr(disc_loss, 'item'):
            disc_loss = disc_loss.item()
        
        # Calculate weighted discriminator loss
        weighted_disc_loss = self.disc_loss_weight * self.disc_warmup_weight * disc_loss
        logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                   f"Loss={main_loss:.6f}, Rec={rec_loss:.6f}, KL={kl_loss:.6f}, "
                   f"Disc={disc_loss:.6f} (weighted: {weighted_disc_loss:.6f})")

    def _log_validation_progress(self, batch_count, losses):
        """Log validation progress to stdout."""
        # Get the main loss value
        main_loss = losses.get('loss', 0)
        if hasattr(main_loss, 'item'):
            main_loss = main_loss.item()
        
        # Get other important metrics
        kl_loss = losses.get('kl_loss', 0)
        if hasattr(kl_loss, 'item'):
            kl_loss = kl_loss.item()
        
        rec_loss = losses.get('rec_loss', 0)
        if hasattr(rec_loss, 'item'):
            rec_loss = rec_loss.item()
        
        disc_loss = losses.get('disc_loss', 0)
        if hasattr(disc_loss, 'item'):
            disc_loss = disc_loss.item()
        
        # Calculate weighted discriminator loss
        weighted_disc_loss = self.disc_loss_weight * self.disc_warmup_weight * disc_loss
        logger.info(f"Validation Epoch {self.current_epoch}, Batch {batch_count}: "
                   f"Loss={main_loss:.6f}, Rec={rec_loss:.6f}, KL={kl_loss:.6f}, "
                   f"Disc={disc_loss:.6f} (weighted: {weighted_disc_loss:.6f})")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean: bool = True, return_counts: bool = False):
        inference_outputs, generative_outputs = self(batch, encode=True, predict_mode=True)
        if return_counts:
            if "cell_idx" in batch:
                return torch.cat((generative_outputs["counts_pred"], torch.unsqueeze(batch["cell_idx"], 1)), 1) 
        else:
            qz_m = inference_outputs["qz_m"]
            z = qz_m if give_mean else inference_outputs["z"]
            z = z.double()
            if "cell_idx" in batch:
                return torch.cat((z, torch.unsqueeze(batch["cell_idx"], 1)), 1)
            elif "locate" in batch:
                return torch.cat((z, batch["locate"]), 1)
            else:
                return z

    def configure_optimizers(self,):
        if self.training_args.get("train_library"):
            g_params = itertools.chain(
                self.vae.z_encoder.parameters(),
                self.vae.l_encoder.parameters(),
                self.vae.decoder.parameters()
            )
        else:
            g_params = itertools.chain(
                self.vae.z_encoder.parameters(),
                self.vae.decoder.parameters()
            )
        vae_optimizer = torch.optim.Adam(g_params, **self.training_args["vae_optimizer"])
        vae_config = {"optimizer": vae_optimizer}
        if self.training_args.get("reduce_lr_on_plateau"):
            vae_scheduler = ReduceLROnPlateau(
                vae_optimizer,
                mode='min',           
                factor=0.5,           
                patience=4,           
                min_lr=1e-6
            )
            vae_config["lr_scheduler"] = {
                "scheduler": vae_scheduler,
                "monitor": "val_loss/loss",
            }
        if self.training_args.get("train_adversarial"):
            params = [x.parameters() for x in self.vae.x_predictors]
            params += [z.parameters() for z in self.vae.z_predictors]
            p_params = itertools.chain(*params)
            discriminator_optimizer = torch.optim.Adam(p_params, **self.training_args["discriminator_optimizer"])        
            discriminator_config = {"optimizer": discriminator_optimizer}
            if self.training_args.get("reduce_lr_on_plateau"):
                disc_scheduler = ReduceLROnPlateau(
                    discriminator_optimizer,
                    mode='min',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                )
                discriminator_config["lr_scheduler"] = {
                    "scheduler": disc_scheduler,
                    "monitor": "val_loss/disc_loss",
                }
            return [vae_config, discriminator_config]
        else:
            return vae_config
