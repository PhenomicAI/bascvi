from typing import Dict, Union, Tuple, List
import torch
import torch.nn as nn
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal

from bascvi.model.bdecoder import BDecoder
from bascvi.model.bencoder import BEncoder
from bascvi.model.encoder import Encoder

from bascvi.model.distributions import ZeroInflatedNegativeBinomial

from torch.nn import CosineSimilarity


class BAScVI(nn.Module):

    """Variational auto-encoder model.

    This is an implementation of the scVI model descibed in [Lopez18]_

    Parameters
    ----------
    n_input : int
        Number of input genes
    n_batch : int, optional
        Number of batches, if 0, no batch correction is performed, by default 0
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
        n_batch: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        normalize_total: bool = True,
        scaling_factor: float = 10000.0,
        init_weights: bool = True,
        batch_emb_dim: int = 0,  # default 0, 10,
        use_library = True,
        use_batch_encoder = True,
        use_zinb = True,
        macrogene_matrix = None,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.log_variational = log_variational
        self.normalize_total = normalize_total
        self.scaling_factor = scaling_factor
        self.use_library = use_library
        self.use_zinb = use_zinb

        self.cs_loss = CosineSimilarity()

        self.batch_emb_dim = batch_emb_dim

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        if macrogene_matrix:
            self.macrogene_matrix = torch.nn.Parameter(macrogene_matrix)


        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input if self.macrogene_matrix is None else self.macrogene_matrix.shape[1]
        
        self.z_encoder = BEncoder(
            n_input=n_input_encoder,
            n_batch=n_batch,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        if self.use_library:
            self.l_encoder = Encoder(
                n_input=n_input_encoder,
                n_batch=n_batch,
                n_output=1,
                n_layers=1,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )

        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = BDecoder(
            n_input=n_input_decoder,
            n_batch=n_batch,
            n_output=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            macrogene_dim=macrogene_matrix.shape[1] if macrogene_matrix is not None else None,
        )
        
        self.z_predictor = BPredictor(
            n_input=n_hidden,
            n_batch=n_batch,
        )
        self.x_predictor = BPredictor(
            n_input=n_hidden,
            n_batch=n_batch,
        )
        
        self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        if init_weights:
            self.apply(self.init_weights)


    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def inference(self, x, batch_emb):
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
        
        qz_m, qz_v, z, x_pred = self.z_encoder(x, batch_emb)

        x_pred = self.x_predictor(x_pred)

        if self.use_library:
            ql_m, ql_v, library_encoded = self.l_encoder(x, batch_emb)
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library_encoded, x_pred=x_pred)
            
        else:
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, x_pred=x_pred)

        return outputs

    def generative(self, z, batch_emb, library=None):
        """Run the generative step i.e forward pass of decoder of VAE, computes
        the parameters associated with the likelihood of the data.

        This is typically written as `p(x|z)`.
        """
        decoder_input = z
        px_scale, px_rate, px_dropout,z_pred = self.decoder(
            decoder_input,
            batch_emb,
            library=library,
            macrogene_matrix=self.macrogene_matrix,
        )

        px_r = self.px_r
        px_r = torch.exp(px_r)       

        z_pred = self.z_predictor(z_pred)
        
        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout, z_pred=z_pred)

    def forward(
        self, batch: dict, kl_weight: float = 1.0, disc_loss_weight: float = 10.0, disc_warmup_weight: float = 1.0, kl_loss_weight: float = 1.0, compute_loss: bool = True, encode: bool = False, optimizer_idx=0, predict_mode=False
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
            x_ = torch.log(1 + self.scaling_factor * x / x.sum(dim=1, keepdim=True))

        if self.macrogene_matrix:
            x_ = nn.functional.linear(x_, self.macrogenes)

        inference_outputs = self.inference(x_, batch_vec)

        if encode:
            return inference_outputs

        z = inference_outputs["z"]

        if self.use_library:
            library = inference_outputs["library"]
            generative_outputs = self.generative(z, batch_vec, library=library)
        else:
            generative_outputs = self.generative(z, batch_vec)


        if compute_loss:
            losses = self.loss(
                x,
                local_l_mean,
                local_l_var,
                inference_outputs,
                generative_outputs,
                batch_vecs=[modality_vec, study_vec, sample_vec],
                feature_presence_mask=batch["feature_presence_mask"],
                kl_weight=kl_weight,
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
        kl_weight: float = 1.0,
        disc_loss_weight: float = 10.0,
        disc_warmup_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
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

                weighted_kl_local = kl_divergence_z + kl_weight * kl_divergence_l
            
            
            z_pred = generative_outputs["z_pred"]
            x_pred = inference_outputs["x_pred"]

            batch_vec = torch.cat(batch_vecs, dim=1)
            
            # batch adversarial cost z_pred is from the decoder, x_pred is from the encoder. we want to remove batch from both
            # TODO: question: do we want to remove modality from decoder output? we might want to be able to translate between modalities
            disc_loss = disc_loss_weight * (self.loss_bce(z_pred, batch_vec.float()) + self.loss_bce(x_pred, batch_vec.float()))

            # ensure we give equal weight to each batch dimension (modality, study, sample)
            disc_loss_modality = torch.mean(torch.mean(disc_loss[ : , :batch_vecs[0].shape[0]], dim=1))
            disc_loss_study = torch.mean(torch.mean(disc_loss[ : , batch_vecs[0].shape[0]:batch_vecs[0].shape[0] + batch_vecs[1].shape[0]], dim=1))
            disc_loss_sample = torch.mean(torch.mean(disc_loss[ : , batch_vecs[0].shape[0] + batch_vecs[1].shape[0]:], dim=1))

            disc_loss_reduced = torch.mean(torch.mean(disc_loss, dim=1))

            reconst_loss = torch.mean(reconst_loss)
            weighted_kl_local = kl_loss_weight * (torch.mean(weighted_kl_local))

            loss = reconst_loss + weighted_kl_local - disc_warmup_weight * disc_loss_reduced 

            return {"loss": loss, "rec_loss": reconst_loss.detach(), "kl_local": weighted_kl_local.detach(), "disc_loss": disc_loss_reduced.detach(), "disc_loss_modality": disc_loss_modality.detach(), "disc_loss_study": disc_loss_study.detach(), "disc_loss_sample": disc_loss_sample.detach()}
            
        if optimizer_idx == 1:

            batch_vec = torch.cat(batch_vecs, dim=1)
            
            z_pred = generative_outputs["z_pred"]
            x_pred = inference_outputs["x_pred"]
            
            # batch adversarial cost z_pred is from the decoder, x_pred is from the encoder. we want to remove batch from both
            # TODO: question: do we want to remove modality from decoder output? we might want to be able to translate between modalities
            disc_loss = disc_loss_weight * (self.loss_bce(z_pred, batch_vec.float()) + self.loss_bce(x_pred, batch_vec.float()))

            # ensure we give equal weight to each batch dimension (modality, study, sample)
            disc_loss_modality = torch.mean(torch.mean(disc_loss[ : , :batch_vecs[0].shape[0]], dim=1))
            disc_loss_study = torch.mean(torch.mean(disc_loss[ : , batch_vecs[0].shape[0]:batch_vecs[0].shape[0] + batch_vecs[1].shape[0]], dim=1))
            disc_loss_sample = torch.mean(torch.mean(disc_loss[ : , batch_vecs[0].shape[0] + batch_vecs[1].shape[0]:], dim=1))

            disc_loss_reduced = torch.mean(torch.mean(disc_loss, dim=1))

            return {"loss" : disc_loss_reduced, "rec_loss": 0, "kl_local": 0, "disc_loss": disc_loss_reduced.detach(), "disc_loss_modality": disc_loss_modality.detach(), "disc_loss_study": disc_loss_study.detach(), "disc_loss_sample": disc_loss_sample.detach()}

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, feature_presence_mask) -> torch.Tensor:
        """Computes reconstruction loss."""
        if self.use_zinb:
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x)
            )
            reconst_loss = reconst_loss * feature_presence_mask
        else:
            reconst_loss = 10000 * (1 - self.cs_loss(px_dropout, x))

        reconst_loss = reconst_loss.sum(dim=-1)
        return reconst_loss
        
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
            nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.LeakyReLU(),
            
            nn.Linear(
                    n_hidden,
                    n_batch,
                    ),
            nn.BatchNorm1d(n_batch, eps=1e-5),
            )

    def forward(
        self,
        x: torch.Tensor,
    ):
        
        x = self.predictor(x)
        
        return x






