from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal

from model.decoder import Decoder
from model.encoder import Encoder
from model.distributions import ZeroInflatedNegativeBinomial


class ScVI(nn.Module):
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
        Log(data+1) prior to encoding for numerical stability. Not normalization, by default True
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        normalize_total: bool = True,
        scaling_factor: float = 10000.0,
        init_weights: bool = True,
        batch_emb_dim: int = 0,  # default 0, 10
        use_library = True,
        use_batch_encoder = True,
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
        self.batch_emb_dim = batch_emb_dim
        self.use_library = use_library
        self.use_batch_encoder = use_batch_encoder

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input

        self.z_encoder = Encoder(
            n_input=n_input_encoder,
            n_batch=n_batch,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        if self.use_library:
            # l encoder goes from n_input-dimensional data to 1-d library size
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
        self.decoder = Decoder(
            n_input=n_input_decoder,
            n_batch=n_batch,
            n_output=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
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
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)
        elif self.normalize_total:
            x_ = torch.log(1 + self.scaling_factor * x_ / x_.sum(dim=1, keepdim=True))

        encoder_input = x_
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_emb,use_batch_encoder=self.use_batch_encoder)
        
        if self.use_library:
            ql_m, ql_v, library_encoded = self.l_encoder(encoder_input, batch_emb)
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library_encoded)
            
        else:
            outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v)
            
        return outputs

    def generative(self, z, batch_emb, library=None):
        """Run the generative step i.e forward pass of decoder of VAE, computes
        the parameters associated with the likelihood of the data.

        This is typically written as `p(x|z)`.
        """
        decoder_input = z
        px_scale, px_rate, px_dropout = self.decoder(
            decoder_input,
            batch_emb,
            library=library,
        )

        px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def forward(self, batch, kl_weight=1.0, compute_loss=True, encode=False, optimizer_idx=None):
    
        x = batch["x"]
        batch_emb = batch["batch_emb"]
        local_l_mean = batch["local_l_mean"]
        local_l_var = batch["local_l_var"]
        inference_outputs = self.inference(x, batch_emb)

        if encode:
            return inference_outputs

        z = inference_outputs["z"]
        
        if self.use_library:
            library = inference_outputs["library"]
            generative_outputs = self.generative(z, batch_emb, library=library)
        else:
            generative_outputs = self.generative(z, batch_emb)
            
        if compute_loss:
            losses = self.loss(
                x,
                local_l_mean,
                local_l_var,
                inference_outputs,
                generative_outputs,
                kl_weight=kl_weight,
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
        kl_weight: float = 1.0,
    ) -> Dict:
    
        """Compute the loss for a minibatch of data.

        This function uses the outputs of the inference and generative
        functions to compute a loss. This many optionally include other
        penalty terms, which should be computed here.
        """

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v + 1e-6)), Normal(mean, scale)).sum(dim=1)
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        
        weighted_kl_local = kl_divergence_z
        
        if self.use_library:
        
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v + 1e-6)),
                Normal(local_l_mean, torch.sqrt(local_l_var + 1e-6)),
            ).sum(dim=1)

            weighted_kl_local = kl_divergence_z + kl_weight * kl_divergence_l

        
        loss = torch.mean(reconst_loss + weighted_kl_local)

        return {"loss": loss, "rec_loss": reconst_loss.sum().detach(), "kl_local": weighted_kl_local.sum().detach()}

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout) -> torch.Tensor:
        """Computes reconstruction loss."""

        reconst_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x).sum(dim=-1)
        )
        return reconst_loss
