import logging
import itertools
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from bascvi.utils.utils import umap_calc_and_save_html
import os, shutil

from bascvi.datamodule.soma.soma_helpers import open_soma_experiment

import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BAScVITrainer(pl.LightningModule):
    """Lightning module to train scvi-vae model.
    """

    def __init__(
        self,
        root_dir,
        model_args: dict = {},
        training_args: dict = {},
        callbacks_args: dict = {},
        module_name: str = "",
        class_name: str = "",
    ):
        super().__init__()
        # save hyperparameters in hparams.yaml file
        self.save_hyperparameters(ignore="datamodule")

        self.root_dir = root_dir
        self.valid_counter = 0

        self.model_args = model_args
        self.training_args = training_args
        self.callbacks_args = callbacks_args
        self.n_epochs_kl_warmup = training_args.get("n_epochs_kl_warmup")
        self.disc_loss_weight = training_args.get("disc_loss_weight")
        self.kl_loss_weight = training_args.get("kl_loss_weight")
        self.n_steps_kl_warmup = training_args.get("n_steps_kl_warmup")
        self.n_epochs_discriminator_warmup = training_args.get("n_epochs_discriminator_warmup")
        self.n_steps_discriminator_warmup = training_args.get("n_steps_discriminator_warmup")
        self.use_library = training_args.get("use_library")
        self.save_validation_umaps = training_args.get("save_validation_umaps")


        # neat way to dynamically import classes
        # __import__ method used to fetch module
        
        module = __import__("bascvi.model", globals(), locals(), [module_name], 0)
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
                patience=self.callbacks_args["early_stopping"]["patience"]*2,
                mode=self.callbacks_args["early_stopping"]["mode"]
            )
            self.callbacks.append(self.early_stop_callback)

    def forward(self, batch, **kwargs):
        # in lightning, forward defines the prediction/inference actions
        return self.vae(batch, **kwargs)

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        epoch_criterion = self.n_epochs_kl_warmup is not None
        step_criterion = self.n_steps_kl_warmup is not None
        if epoch_criterion:
            kl_weight = min(1.0, self.current_epoch / self.n_epochs_kl_warmup)
        elif step_criterion:
            kl_weight = min(1.0, self.global_step / self.n_steps_kl_warmup)
        else:
            kl_weight = 1.0
        return kl_weight
    
    @property
    def disc_warmup_weight(self):
        """Scaling factor on KL divergence during training."""
        epoch_criterion = self.n_epochs_discriminator_warmup is not None
        step_criterion = self.n_steps_discriminator_warmup is not None
        if epoch_criterion:
            disc_warmup_weight = min(1.0, self.current_epoch / self.n_epochs_discriminator_warmup)
        elif step_criterion:
            disc_warmup_weight = min(1.0, self.global_step / self.n_steps_discriminator_warmup)
        else:
            disc_warmup_weight = 1.0
        return disc_warmup_weight
        
    def training_step(self, batch, batch_idx):
        
        if self.training_args.get("train_adversarial"):
            g_opt, d_opt = self.optimizers()

            # do not remove, skips over small minibatches
            if batch["x"].shape[0] < 3:
                return None
            
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            g_opt.step()

            d_opt.zero_grad()
            _, _, d_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=1)
            self.manual_backward(d_losses['loss'])
            d_opt.step()

            self.log("train_loss", g_losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_rec_loss", g_losses['rec_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_kl_loss", g_losses['kl_local'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_disc_loss", g_losses['disc_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            g_opt = self.optimizers()

            # do not remove, skips over small minibatches
            if batch["x"].shape[0] < 3:
                return None
            
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            g_opt.step()

            self.log("train_loss", g_losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_rec_loss", g_losses['rec_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_kl_loss", g_losses['kl_local'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return g_losses

    def validation_step(self, batch, batch_idx):
        encoder_outputs, _, scvi_loss = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
        
        metrics_to_log = {
            "val_loss": scvi_loss['loss'],
            "val_rec_loss": scvi_loss['rec_loss'],
            "val_disc_loss": scvi_loss['disc_loss'],
            "val_kl_loss": scvi_loss['kl_local'],
        }

        self.log("val_loss", scvi_loss['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rec_loss", scvi_loss['rec_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_disc_loss", scvi_loss['disc_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_kl_loss", scvi_loss['kl_local'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

        z = qz_m

        self.validation_step_outputs.append(torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1))

        wandb.log(metrics_to_log)
        
        return scvi_loss

    def on_validation_epoch_end(self):
        metrics_to_log = {}

        if self.save_validation_umaps:
            logger.info("Running validation UMAP...")
            embeddings = torch.cat(self.validation_step_outputs, dim=0).detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1])[:-1]] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
            # if self.obs_df = None:
            #     with open_soma_experiment(self.datamodule.soma_experiment_uri) as soma_experiment:
            #         self.obs_df = soma_experiment.obs.read(
            #                             column_names=("soma_joinid", "standard_true_celltype", "sample_name", "study_name"),
            #                         ).concat().to_pandas()
            embeddings_df = embeddings_df.set_index("soma_joinid").join(self.datamodule.obs_df.set_index("soma_joinid"))
            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            os.makedirs(save_dir, exist_ok=True)
            embeddings_df, fig_path_dict = umap_calc_and_save_html(embeddings_df, emb_columns, save_dir, color_by=["standard_true_celltype", "study_name", "batch_name", "tissue_primary"])

            for key, fig_path in fig_path_dict.items():
                metrics_to_log[key] = wandb.Image(fig_path, caption=key)
            
            self.valid_counter += 1

            self.validation_step_outputs.clear()
        else:
            pass

        wandb.log(metrics_to_log)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean: bool = True):
        encoder_outputs = self(batch, encode=True)
        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

        if give_mean:
            z = qz_m

        # print('z:', z.shape)
        # print('soma_joinid:', batch["soma_joinid"].shape)
        # # print unique soma_joinid
        # print('unique:', len(batch["soma_joinid"].unique()))

        return torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)

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
        
        optimizer = torch.optim.Adam(g_params, **self.training_args["optimizer"])
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
                "monitor": self.training_args["lr_scheduler_metric"],
            }
        
        elif self.training_args.get("step_lr_scheduler"):
            scheduler = StepLR(optimizer, **self.training_args.get("step_lr_scheduler"))
            config["lr_scheduler"] = scheduler
        
        if self.training_args.get("train_adversarial"):
            print("Adversarial Training: True")
            # Discriminator Optimizer
            
            p_params = itertools.chain(
                self.vae.x_predictor.parameters(),
                self.vae.z_predictor.parameters(),
                #self.vae.mu_predictor.parameters(), uncomment for ablation studies
                #self.vae.emb_predictor.parameters() uncomment for ablation studies
                )
              
            p_opt = torch.optim.Adam(p_params, lr=1e-2, eps=1e-8)        
            plr_scheduler = StepLR(p_opt, **self.training_args.get("step_lr_scheduler"))        
            
            return [config,{"optimizer": p_opt, "lr_scheduler":plr_scheduler}]
            
        else:
            print("Adversarial Training: False")
            return config
                
