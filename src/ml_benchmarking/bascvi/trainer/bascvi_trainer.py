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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ml_benchmarking.bascvi.datamodule.soma.soma_helpers import open_soma_experiment
from ml_benchmarking.bascvi.utils.utils import umap_calc_and_save_html, calc_kni_score, calc_rbni_score
from ml_benchmarking.bascvi.utils.protein_embeddings import get_stacked_protein_embeddings_matrix, get_centroid_distance_matrix


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
        n_batch: int = None,
        use_macrogenes: bool = False,
        macrogene_method: str = "concat",
        macrogene_embedding_model: str = "ESM2",
        macrogene_species_list: list = ['human', 'mouse'],
        macrogene_matrix_path: str = None,
        freeze_macrogene_matrix: bool = True,
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
        if n_batch is not None:
            self.model_args["n_batch"] = n_batch
            
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
        self.run_metrics = training_args.get("run_metrics", False)

        self.gene_list = gene_list

        self.use_macrogenes = use_macrogenes
        self.macrogene_method = macrogene_method
        self.macrogene_embedding_model = macrogene_embedding_model
        self.macrogene_matrix_path = macrogene_matrix_path
        self.freeze_macrogene_matrix = freeze_macrogene_matrix

        self.macrogene_species_list = macrogene_species_list

        # neat way to dynamically import classes
        # __import__ method used to fetch module
        
        module = __import__("ml_benchmarking.bascvi.model", globals(), locals(), [module_name], 0)
        # getting attribute by getattr() method
        Vae = getattr(module, class_name)

        if self.use_macrogenes:
            if self.macrogene_method == "concat_norm":
                macrogene_matrix = get_stacked_protein_embeddings_matrix(f"/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/{self.macrogene_embedding_model}", gene_list=self.gene_list, species_list=self.macrogene_species_list)
                # scale macrogene matrix
                # scaler = StandardScaler()
                # macrogene_matrix = scaler.fit_transform(macrogene_matrix)
                # normalize macrogene matrix dividing by sum of each row
                macrogene_matrix = macrogene_matrix / macrogene_matrix.sum(axis=1, keepdims=True)
            
            elif self.macrogene_method == "concat":
                macrogene_matrix = get_stacked_protein_embeddings_matrix(f"/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/{self.macrogene_embedding_model}", gene_list=self.gene_list, species_list=self.macrogene_species_list)
            
            elif self.macrogene_method == "saturn":
                if os.path.isfile(self.macrogene_matrix_path):
                    macrogene_matrix = np.load(self.macrogene_matrix_path)
                else:
                    macrogene_matrix = get_centroid_distance_matrix(f"/home/ubuntu/paper_repo/bascvi/data/gene_embeddings/{self.macrogene_embedding_model}", gene_list=self.gene_list, species_list=self.macrogene_species_list, num_clusters=10000)
            
            elif self.macrogene_method == "ortholog":
                macrogene_matrix = np.load("/home/ubuntu/paper_repo/bascvi/data/ortho_gene_matrix.npy")



            model_args["macrogene_matrix"] = torch.from_numpy(macrogene_matrix).float()
            model_args["freeze_macrogene_matrix"] = self.freeze_macrogene_matrix

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
                patience=self.callbacks_args["early_stopping"]["patience"],
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
        cyclic_criterion = self.training_args.get("cyclic_kl_period")
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

        epoch_criterion = self.n_epochs_discriminator_warmup is not None
        step_criterion = self.n_steps_discriminator_warmup is not None
        exponential_criterion = self.training_args.get("exponential_disc_warmup")
        if epoch_criterion:
            if exponential_criterion:
                disc_warmup_weight = min(1.0, exp_warmup_scale(self.current_epoch, self.n_epochs_discriminator_warmup))
            else:
                disc_warmup_weight = min(1.0, self.current_epoch / self.n_epochs_discriminator_warmup)
        elif step_criterion:
            if exponential_criterion:
                disc_warmup_weight = min(1.0, exp_warmup_scale(self.global_step, self.n_steps_discriminator_warmup))
            else:
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

            # add train to log_dict and log
            g_losses = {f"train_{k}": v for k, v in g_losses.items()}
            self.log_dict(g_losses)

            d_opt.zero_grad()
            _, _, d_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=1)
            self.manual_backward(d_losses['loss'])
            d_opt.step()

        else:
            g_opt = self.optimizers()

            # do not remove, skips over small minibatches
            if batch["x"].shape[0] < 3:
                return None
            
            g_opt.zero_grad()
            _, _, g_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
            self.manual_backward(g_losses['loss'])
            g_opt.step()

            # add train to log_dict and log
            g_losses = {f"train_{k}": v for k, v in g_losses.items()}

            g_losses["disc_warmup_weight"] = self.disc_warmup_weight
            
            self.log_dict(g_losses)

        return g_losses

    def validation_step(self, batch, batch_idx):
        encoder_outputs, _, g_losses = self.forward(batch, kl_weight=self.kl_weight, disc_loss_weight=self.disc_loss_weight, disc_warmup_weight=self.disc_warmup_weight, kl_loss_weight=self.kl_loss_weight, optimizer_idx=0)
        
        g_losses = {f"val_{k}": v for k, v in g_losses.items()}
        self.log_dict(g_losses)

        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

        z = qz_m


        # Important: make z float64 dtype to concat properly with soma_joinid
        z = z.double()

        emb_output = torch.cat((z, torch.unsqueeze(batch["soma_joinid"], 1)), 1)


        self.validation_step_outputs.append(emb_output)
        
        return g_losses

    def on_validation_epoch_end(self):
        metrics_to_log = {}

        if self.save_validation_umaps:
            logger.info("Running validation UMAP...")
            embeddings = torch.cat(self.validation_step_outputs, dim=0).double().detach().cpu().numpy()
            emb_columns = ["embedding_" + str(i) for i in range(embeddings.shape[1] - 1)] 
            embeddings_df = pd.DataFrame(data=embeddings, columns=emb_columns + ["soma_joinid"])
           
            save_dir = os.path.join(self.root_dir, "validation_umaps", str(self.valid_counter))
            os.makedirs(save_dir, exist_ok=True)

            obs_columns = ["standard_true_celltype", "study_name", "sample_name"]
            if self.use_macrogenes:
                obs_columns.append("species") 
                pass

            if self.obs_df is None:
                with open_soma_experiment(self.soma_experiment_uri) as soma_experiment:
                    self.obs_df = soma_experiment.obs.read(column_names=['soma_joinid'] + obs_columns).concat().to_pandas()
                    self.obs_df = self.obs_df.set_index("soma_joinid")
            
            if "species" not in self.obs_df.columns:
                print("Adding species column to obs, assuming human data")
                self.obs_df["species"] = "human"

                # assign species based on study_name
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_m-"), "species"] = "mouse"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_r-"), "species"] = "rat"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_l-"), "species"] = "lemur"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_c-"), "species"] = "macaque"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_f-"), "species"] = "fly"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_a-"), "species"] = "axolotl"
                self.obs_df.loc[self.obs_df["study_name"].str.contains("_z-"), "species"] = "zebrafish"

                obs_columns.append("species")



            
            _, fig_path_dict = umap_calc_and_save_html(embeddings_df.set_index("soma_joinid").join(self.obs_df, how="inner").reset_index(), emb_columns, save_dir, obs_columns, max_cells=100000)

            for key, fig_path in fig_path_dict.items():
                metrics_to_log[key] = wandb.Image(fig_path, caption=key)

            if self.run_metrics:
                # run metrics
                metrics_dict = {}
                metrics_dict.update(calc_kni_score(embeddings_df.set_index("soma_joinid")[emb_columns], self.obs_df.loc[embeddings_df.index], batch_col="study_name", n_neighbours=15, max_prop_same_batch=0.8))
                metrics_dict.update(calc_rbni_score(embeddings_df.set_index("soma_joinid")[emb_columns], self.obs_df.loc[embeddings_df.index], batch_col="study_name", radius=1.0, max_prop_same_batch=0.8))

                # add "val_" prefix to keys
                metrics_dict = {f"val_{k}": v for k, v in metrics_dict.items()}
                metrics_to_log.update(metrics_dict)
            
            self.valid_counter += 1

            self.validation_step_outputs.clear()
        else:
            pass

        metrics_to_log["trainer/global_step"] = self.global_step

        wandb.log(metrics_to_log)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, give_mean: bool = True):
        encoder_outputs = self(batch, encode=True, predict_mode=True)
        qz_m = encoder_outputs["qz_m"]
        z = encoder_outputs["z"]

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
            
            return [config,{"optimizer": p_opt, "lr_scheduler": plr_scheduler}]
            
        else:
            print("Adversarial Training: False")
            return config
