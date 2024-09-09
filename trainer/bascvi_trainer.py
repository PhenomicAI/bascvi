import logging
import itertools
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BAScVITrainer(pl.LightningModule):
    """Lightning module to train scvi-vae model.
    """

    def __init__(
        self,
        n_genes: int,
        n_batches: int,
        model_args: dict = {},
        training_args: dict = {},
        callbacks_args: dict = {},
        module_name: str = "",
        class_name: str = "",
    ):
        super().__init__()
        # save hyperparameters in hparams.yaml file
        self.save_hyperparameters()

        self.model_args = model_args
        self.training_args = training_args
        self.callbacks_args = callbacks_args
        self.n_epochs_kl_warmup = training_args.get("n_epochs_kl_warmup")
        self.n_steps_kl_warmup = training_args.get("n_steps_kl_warmup")

        # neat way to dynamically import classes
        # __import__ method used to fetch module
        
        module = __import__("model", globals(), locals(), [module_name], 0)
        # getting attribute by getattr() method
        Vae = getattr(module, class_name)

        self.vae = Vae(n_genes, n_batches, **model_args)

        logger.info(f"{module_name} model args:\n {model_args}")

        self.callbacks = []
        self.configure_callbacks()
        logger.info(f"Initialize {module_name} Trainer.")
        
        self.automatic_optimization = False # PyTorch Lightning 2 method

        self.validation_step_outputs = []

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
            filename="scvi-vae-{epoch:02d}-{elbo_val:.2f}",
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
        
    def training_step(self, batch, batch_idx):
        
        g_opt, d_opt = self.optimizers()

        # do not remove, skips over small minibatches
        if batch["x"].shape[0] < 3:
            return None
        
        g_opt.zero_grad()
        _, _, g_losses = self.forward(batch, kl_weight=self.kl_weight,optimizer_idx=0)
        self.manual_backward(g_losses['loss'])
        g_opt.step()

        d_opt.zero_grad()
        _, _, d_losses = self.forward(batch, kl_weight=self.kl_weight,optimizer_idx=1)
        self.manual_backward(d_losses['loss'])
        d_opt.step()

        self.log("train_loss", g_losses['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return g_losses

    def validation_step(self, batch, batch_idx):
        
        _, _, scvi_loss = self.forward(batch, kl_weight=self.kl_weight)
        self.log("val_loss", scvi_loss['loss'])
        self.validation_step_outputs.append(scvi_loss['loss'])
        
        return scvi_loss

    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean: bool = True):
        encoder_outputs = self(batch, encode=True)
        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

        if give_mean:
            z = qz_m

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
                
