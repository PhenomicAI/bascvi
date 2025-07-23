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

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.utils.utils import umap_calc_and_save_html, calc_kni_score, calc_rbni_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/

def get_kld_cycle(epoch, period=20):
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



class BAScVITrainer(pl.LightningModule):
    """Lightning module to train scvi-vae model.
    """

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
        # save hyperparameters in hparams.yaml file
        self.save_hyperparameters(ignore="datamodule")

        self.root_dir = root_dir
        self.valid_counter = 0

        self.soma_experiment_uri = soma_experiment_uri
        self.obs_df = None

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

        # Add predict_only to model_args
        if predict_only:
            self.model_args["predict_only"] = predict_only

        # neat way to dynamically import classes
        # __import__ method used to fetch module
        
        module = __import__("ml_benchmarking.bascvi.model", globals(), locals(), [module_name], 0)
        # getting attribute by getattr() method
        Vae = getattr(module, class_name)
        self.vae = Vae(**model_args)

        logger.info(f"{module_name} model args:\n {model_args}")

        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize {module_name} Trainer.")
        
        self.automatic_optimization = False # PyTorch Lightning 2 method

        self.validation_step_outputs = []

        if os.path.isdir(os.path.join(self.root_dir, "validation_umaps")):
            shutil.rmtree(os.path.join(self.root_dir, "validation_umaps"))
   
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        """Override to handle predict_only mode."""
        predict_only = kwargs.get('predict_only', False)
        
        # First load with strict=False to avoid the missing keys error
        model = super(BAScVITrainer, cls).load_from_checkpoint(
            checkpoint_path, 
            map_location=map_location, 
            strict=False,
            **kwargs
        )
        
        # Set model to eval mode for prediction
        if predict_only:
            model.eval()
            
        return model

    def configure_callbacks(
        self,
    ):
        # save epoch and elbo_validation in name
        # saves a file like: my/path/scvi-vae-epoch=02-elbo_validation=0.32.ckpt
        # checkpoint_callback will save dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks',
        # 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams_name', 'hyper_parameters'])
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
        # in lightning, forward defines the prediction/inference actions
        return self.vae(batch, **kwargs)

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
            kl_weight = get_kld_cycle(self.current_epoch, period=cyclic_criterion)
        else:
            kl_weight = 1.0

        return kl_weight
    
    @property
    def disc_warmup_weight(self):
        """Scaling factor on discriminator loss during training."""

        def exp_warmup_scale(epoch, max_epochs):
            """Returns a scaling factor between 0.5 and 1.0"""
            alpha = 4
            progress = epoch / max_epochs
            return 0.5 + 0.5 * (1 - np.exp(-alpha * progress))

        def sigmoid_warmup(epoch, max_epochs):
            """Returns a scaling factor between 0.0 and 1.0"""
            beta = 10
            T = 1
            progress = epoch / max_epochs
            return 1 / (1 + np.exp(-beta * (progress - T / 2)))

        epoch_criterion = self.n_epochs_discriminator_warmup is not None
        step_criterion = self.n_steps_discriminator_warmup is not None
        sigmoidal_criterion = self.training_args.get("sigmoidal_disc_warmup")
        if epoch_criterion:
            if sigmoidal_criterion:
                disc_warmup_weight = min(1.0, sigmoid_warmup(self.current_epoch, self.n_epochs_discriminator_warmup))
            else:
                disc_warmup_weight = min(1.0, self.current_epoch / self.n_epochs_discriminator_warmup)
        elif step_criterion:
            if sigmoidal_criterion:
                disc_warmup_weight = min(1.0, sigmoid_warmup(self.global_step, self.n_steps_discriminator_warmup))
            else:
                disc_warmup_weight = min(1.0, self.global_step / self.n_steps_discriminator_warmup)
        else:
            disc_warmup_weight = 1.0
        return disc_warmup_weight
        
    def training_step(self, batch, batch_idx):
        # do not remove, skips over small minibatches
        if batch["x"].shape[0] < 3:
            return None
        
        if self.training_args.get("train_adversarial"):
            g_opt, d_opt = self.optimizers()

            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            # clip gradients
            utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            g_opt.step()

            d_opt.zero_grad()
            _, _, d_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=1)
            self.manual_backward(d_losses['loss'])
            # clip gradients
            utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            d_opt.step()

        else:
            g_opt = self.optimizers()
            
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            # clip gradients
            utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            g_opt.step()


        # add train to log_dict and log
        g_losses = {f"train_loss/{k}": v for k, v in g_losses.items()}
        self.log_dict(g_losses)

        # log trainer metrics
        self.log("trainer/disc_warmup_weight", self.disc_warmup_weight)
        self.log("trainer/kl_warmup_weight", self.kl_warmup_weight)
        self.log("trainer/global_step", self.global_step)
        # self.log("trainer/lr_vae", self.optimizers().param_groups[0]["lr"])
        # if self.training_args.get("train_adversarial"):
        #     self.log("trainer/lr_discriminator", self.optimizers().param_groups[1]["lr"])

        return g_losses

    def validation_step(self, batch, batch_idx):
        encoder_outputs, _, g_losses = self.forward(batch, kl_warmup_weight=self.kl_warmup_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
        
        g_losses = {f"val_loss/{k}": v for k, v in g_losses.items()}
        self.log_dict(g_losses, on_step=False, on_epoch=True)

        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

        z = qz_m


        # Important: make z float64 dtype to concat properly with soma_joinid
        z = z.double()

        emb_output = torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)


        self.validation_step_outputs.append(emb_output)
        
        return g_losses

    def on_validation_epoch_end(self):
        backend = getattr(self.datamodule, "backend", "soma")
        if backend == "soma":
            with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                # ... existing SOMA-specific metrics code ...
                pass  # (keep your current logic here)
        elif backend == "zarr":
            print("Skipping SOMA validation: running with Zarr backend.")
            # Optionally, implement Zarr-based metrics here
        else:
            print("Unknown backend, skipping validation/metrics.")


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean: bool = True, return_counts: bool = False):
        inference_outputs, generative_outputs = self(batch, encode=True, predict_mode=True)

        if return_counts:
            if "soma_joinid" in batch:
                return torch.cat((generative_outputs["counts_pred"], torch.unsqueeze(batch["soma_joinid"], 1)), 1) 

        else:
            qz_m = inference_outputs["qz_m"]
            z = inference_outputs["z"]

            if give_mean:
                z = qz_m

            # Important: make z float64 dtype to concat properly with soma_joinid
            z = z.double()

            if "soma_joinid" in batch:
                # join z with soma_joinid and cell_idx
                return torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)
            elif "locate" in batch:
                return torch.cat((z, batch["locate"]), 1)
            else:
                return z

    def configure_optimizers(self,):
        if self.training_args.get("train_library"):
            print("Library Training: True")
            g_params = itertools.chain(
                self.vae.z_encoder.parameters(),
                self.vae.l_encoder.parameters(),
                self.vae.decoder.parameters()
                )
        else:
            print("Library Training: False")
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
                factor=0.5,           # halve the learning rate
                patience=4,           # wait 4 epochs for improvement
                min_lr=1e-6
            )
            vae_config["lr_scheduler"] = {
                "scheduler": vae_scheduler,
                "monitor": "val_loss/loss",
            }
        

        if self.training_args.get("train_adversarial"):
            print("Adversarial Training: True")
            # Discriminator Optimizer

            params = [x.parameters() for x in self.vae.x_predictors]
            params += [z.parameters() for z in self.vae.z_predictors]
            
            p_params = itertools.chain(
                *params
                )
              
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
            print("Adversarial Training: False")
            return vae_config
