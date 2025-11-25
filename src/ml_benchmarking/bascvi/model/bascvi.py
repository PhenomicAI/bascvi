from typing import List, Dict, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ml_benchmarking.bascvi.model.bdecoder import BDecoder
from ml_benchmarking.bascvi.model.bencoder import BEncoder
from ml_benchmarking.bascvi.model.encoder import Encoder
from ml_benchmarking.bascvi.model.distributions import ZeroInflatedNegativeBinomial

import numpy as np

class BAScVI(nn.Module):

    """Variational auto-encoder model.

    This is an optimized version of the scVI model descibed in [Lopez18] with the addition of a discriminator network to predict the batch labels.

    Parameters
    ----------
    n_input : int
        Number of input genes
    batch_level_sizes : list
        Number of batches
    n_hidden : int, optional
        Number of nodes per hidden layer, by default 128
    n_latent : int, optional
        Dimensionality of the latent space, by default 10
    n_layers : int, optional
        Number of hidden layers used for encoder and decoder NNs, by default 1
    dropout_rate : float, optional
        Dropout rate for neural networks, by default 0.1
    log_variational : bool, optional
        Log(data+1) prior to encoding for numerical stability. Not normalization, by default False
    """

    def __init__(
        self,
        n_input: int,
        batch_level_sizes: List[int],
        n_batch: int = 0,
        n_hidden: int = 512,
        n_latent: int = 10,
        n_layers: int = 4,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        normalize_total: bool = True,
        scaling_factor: float = 10000.0,
        init_weights: bool = True,
        batch_emb_dim: int = 0,  # default 0, 10,
        use_library = True,
        use_batch_encoder = True,
        predict_only = False

    ):
        super().__init__()

        self.n_input = n_input
        
        self.batch_level_sizes = batch_level_sizes
        self.n_batch = np.sum(batch_level_sizes)
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.normalize_total = normalize_total
        self.scaling_factor = scaling_factor
        self.use_library = use_library

        self.batch_emb_dim = batch_emb_dim

        # Initialize px_r close to zero to avoid numerical instability when exp(px_r) is computed
        # This ensures exp(px_r) starts around 1.0 rather than extreme values
        self.px_r = torch.nn.Parameter(torch.zeros(n_input))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input 
        
        self.z_encoder = BEncoder(
            n_input=n_input_encoder,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        if self.use_library:
            self.l_encoder = Encoder(
                n_input=n_input_encoder,
                n_output=1,
                n_layers=1,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )

        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = BDecoder(
            n_input=n_input_decoder,
            n_batch=self.n_batch,
            n_output=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        
        # Only initialize discriminator networks if not in predict_only mode
        if not predict_only:
            self.z_predictors = nn.ModuleList([BPredictor(n_input=n_hidden, n_batch=n_batch, n_hidden=n_hidden) for n_batch in batch_level_sizes])
            self.x_predictors = nn.ModuleList([BPredictor(n_input=n_hidden, n_batch=n_batch, n_hidden=n_hidden) for n_batch in batch_level_sizes])
            self.loss_cce = torch.nn.CrossEntropyLoss()
        else:
            # Create empty placeholders or None for these attributes
            self.z_predictors = None
            self.x_predictors = None
            self.loss_cce = None
            
        if init_weights:
            self.apply(self.init_weights)


    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def inference(self, x):
        """Run the inference step i.e forward pass of encoder of VAE to compute
        variational distribution parameters.

        Parameters
        ----------
        x
            input cell-gene data
        batch_emb
            batch_emb for input data
        Returns
        -------
        outputs : dict
            dictionary of parameters for latent variational distribution
        """
        
        qz_m, qz_v, z, x_pred = self.z_encoder(x)

        x_preds = []

        if hasattr(self, 'x_predictors') and self.x_predictors is not None:
            for i, x_predictor in enumerate(self.x_predictors):
                x_preds.append(x_predictor(x_pred))

        if self.use_library:
            ql_m, ql_v, library_encoded = self.l_encoder(x)
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library_encoded, x_preds=x_preds)
            
        else:
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, x_preds=x_preds)

        return outputs

    def generative(self, z, batch_emb, library=None):
        """Run the generative step i.e forward pass of decoder of VAE, computes
        the parameters associated with the likelihood of the data.

        This is typically written as `p(x|z)`.
        """
        decoder_input = z
        px_scale, px_rate, px_dropout, z_pred = self.decoder(
            decoder_input,
            batch_emb,
            library=library,
        )

        px_r = self.px_r
        px_r = torch.exp(px_r)       

        z_preds = []
        if hasattr(self, 'z_predictors') and self.z_predictors is not None:
            for z_predictor in self.z_predictors:
                z_preds.append(z_predictor(z_pred))

        #counts_pred = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).sample()
        
        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout, z_preds=z_preds) #, counts_pred=counts_pred)

    def forward(
        self, batch: dict, kl_warmup_weight: float = 1.0, 
        disc_loss_weight: float = 10.0, 
        disc_warmup_weight: float = 1.0, 
        kl_loss_weight: float = 1.0, 
        compute_loss: bool = True, 
        encode: bool = False, 
        optimizer_idx=0, 
        predict_mode=False,
        use_zinb: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict],]:
        
        x = batch["x"]

        if not predict_mode:
            modality_vec = batch["modality_vec"]
            study_vec = batch["study_vec"]
            sample_vec = batch["sample_vec"]
            batch_vec = torch.cat([modality_vec, study_vec, sample_vec], dim=1)

            local_l_mean = batch["local_l_mean"]
            local_l_var = batch["local_l_var"]

        else:
            batch_vec = torch.zeros(x.shape[0], self.n_batch).to(x.device)
            modality_vec = torch.zeros(x.shape[0], 1).to(x.device)
            study_vec = torch.zeros(x.shape[0], 1).to(x.device)
            sample_vec = torch.zeros(x.shape[0], 1).to(x.device)

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x)

        elif self.normalize_total:
            x_ = torch.log(1 + self.scaling_factor * x / (x.sum(dim=1, keepdim=True) + 1e-6))

        inference_outputs = self.inference(x_)

        if encode:
            return inference_outputs, None

        z = inference_outputs["z"]

        if self.use_library:
            library = inference_outputs["library"]
            generative_outputs = self.generative(z, batch_vec, library=library)
        else:
            generative_outputs = self.generative(z, batch_vec)


        if compute_loss and not predict_mode:
            losses = self.loss(
                x,
                local_l_mean,
                local_l_var,
                inference_outputs,
                generative_outputs,
                batch_vecs=[modality_vec, study_vec, sample_vec],
                feature_presence_mask=batch["feature_presence_mask"],
                kl_warmup_weight=kl_warmup_weight,
                disc_loss_weight=disc_loss_weight,
                disc_warmup_weight=disc_warmup_weight,
                kl_loss_weight=kl_loss_weight,
                optimizer_idx=optimizer_idx
            )
            return inference_outputs, generative_outputs, losses
        else:
            return inference_outputs, generative_outputs

    def loss(
        self,
        x,
        local_l_mean,
        local_l_var,
        inference_outputs,
        generative_outputs,
        batch_vecs: List[torch.Tensor],
        feature_presence_mask,
        disc_loss_weight: float = 1.0,
        disc_warmup_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        kl_warmup_weight: float = 1.0,
        optimizer_idx=0,

    ) -> Dict:
    
        """Compute the loss for a minibatch of data.

        This function uses the outputs of the inference and generative
        functions to compute a loss. This many optionally include other
        penalty terms, which should be computed here.
        """

        if optimizer_idx == 0:
        
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            px_rate = generative_outputs["px_rate"]
            px_r = generative_outputs["px_r"]
            px_dropout = generative_outputs["px_dropout"]
            
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)
            
            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v + 1e-6)), Normal(mean, scale)).sum(dim=1)

            reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout, feature_presence_mask)
            weighted_kl_local = kl_divergence_z
        
            if self.use_library:
        
                ql_m = inference_outputs["ql_m"]
                ql_v = inference_outputs["ql_v"]
            
                kl_divergence_l = kl(
                    Normal(ql_m, torch.sqrt(ql_v + 1e-6)),
                    Normal(local_l_mean, torch.sqrt(local_l_var + 1e-6)),
                ).sum(dim=1)

                weighted_kl_local = kl_divergence_z + kl_divergence_l
            
            z_preds = generative_outputs["z_preds"]
            x_preds = inference_outputs["x_preds"]

            z_disc_loss_reduced, z_disc_losses = self.get_disc_loss(z_preds, batch_vecs)
            x_disc_loss_reduced, x_disc_losses = self.get_disc_loss(x_preds, batch_vecs)

            disc_loss_reduced = z_disc_loss_reduced + x_disc_loss_reduced

            reconst_loss = torch.mean(reconst_loss)
            weighted_kl_local = torch.mean(weighted_kl_local)

            loss = reconst_loss + kl_warmup_weight * kl_loss_weight * weighted_kl_local - disc_loss_weight * disc_warmup_weight * disc_loss_reduced 

            loss_dict = {
                "loss": loss, 
                "rec_loss": reconst_loss.detach(), 
                "kl_loss": weighted_kl_local.detach(), 
                "kl_normal": torch.mean(kl_divergence_z).detach(),
                "kl_library": torch.mean(kl_divergence_l).detach() if self.use_library else 0,
                "disc_loss": disc_loss_reduced.detach(), 
                }

            for i, z_disc_loss in enumerate(z_disc_losses):
                loss_dict[f"disc_loss_z_{i}"] = z_disc_loss.detach()

            for i, x_disc_loss in enumerate(x_disc_losses):
                loss_dict[f"disc_loss_x_{i}"] = x_disc_loss.detach()
            
        elif optimizer_idx == 1:
            
            z_preds = generative_outputs["z_preds"]
            x_preds = inference_outputs["x_preds"]

            z_disc_loss_reduced, z_disc_losses = self.get_disc_loss(z_preds, batch_vecs)
            x_disc_loss_reduced, x_disc_losses = self.get_disc_loss(x_preds, batch_vecs)

            disc_loss_reduced = z_disc_loss_reduced + x_disc_loss_reduced

            loss_dict = {
                "loss": disc_loss_reduced, 
                "rec_loss": 0, 
                "kl_loss": 0, 
                "kl_normal": 0,
                "kl_library": 0,
                "disc_loss": disc_loss_reduced.detach(), 
                }

            for i, z_disc_loss in enumerate(z_disc_losses):
                loss_dict[f"disc_loss_z_{i}"] = z_disc_loss.detach()

            for i, x_disc_loss in enumerate(x_disc_losses):
                loss_dict[f"disc_loss_x_{i}"] = x_disc_loss.detach()
        else:
            # Fallback for unexpected optimizer_idx values
            raise ValueError(f"Unexpected optimizer_idx: {optimizer_idx}. Expected 0 or 1.")

        return loss_dict
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, feature_presence_mask) -> torch.Tensor:
        """Computes reconstruction loss."""
        
        reconst_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x)
        )
        
        reconst_loss = reconst_loss * feature_presence_mask
        reconst_loss = reconst_loss.sum(dim=-1)
        
        return reconst_loss
    
    def get_disc_loss(self, preds, batch_vecs):

         # Skip if we're in predict_only mode or loss_cce is None

        if not hasattr(self, 'loss_cce') or self.loss_cce is None:
            return torch.tensor(0.0, device=preds[0].device if preds else None), []
        
        disc_losses = []

        for i, pred in enumerate(preds):
            disc_losses.append(self.loss_cce(pred, batch_vecs[i].argmax(dim=1)))

        # TODO: add weights for the different losses?
        disc_loss_reduced = torch.mean(torch.stack(disc_losses))

        return disc_loss_reduced, disc_losses

class BPredictor(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_hidden: int = 128,
    ):
        super().__init__()
        
        # prediction layer
        
        self.predictor = nn.Sequential(
            nn.Linear(
                    n_input,
                    n_hidden,
                    ),
            nn.LayerNorm(n_hidden),
            nn.LeakyReLU(),
            
            nn.Linear(
                    n_hidden,
                    n_batch,
                    ),
            nn.LayerNorm(n_batch),
            )

    def forward(
        self,
        x: torch.Tensor,
    ):
        
        x = self.predictor(x)
        
        return x






