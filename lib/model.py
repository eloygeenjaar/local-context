import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch import optim, nn
from torch import distributions as D
from torch.nn import functional as F
from scipy.stats import pearsonr
# Import all kernel functions
from .modules import EncoderLocal, EncoderGlobal, WindowDecoder
from sklearn.linear_model import LinearRegression
from .utils import mean_corr_coef as mcc
from numbers import Number


class BaseModel(pl.LightningModule):
    def __init__(self, input_size, time_length, local_size, global_size, window_size=20,
                 kernel='cauchy', beta=1., lamda=1., M=1, sigma=1.0,
                 length_scale=1.0, kernel_scales=1, p=100):
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.global_encoder = EncoderGlobal(input_size, global_size, 64, 3, window_size, 3)
        self.local_encoder = EncoderLocal(input_size, local_size, window_size, 64, 3)
        self.input_size = input_size
        self.window_size = window_size
        self.decoder = WindowDecoder(input_size, local_size, global_size, window_size, 64, 4)
        self.beta = beta
        self.lamda = lamda
        # encoder params
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x):
        batch_size, num_timesteps, _ = x.size()
        h_l, global_dist = self.global_encoder(x, self.training)
        h_g, local_dist = self.local_encoder(x, window_size=self.window_size)
        z_t = local_dist.rsample()
        z_g = global_dist.rsample()
        p_x_hat = self.decoder(z_t, z_g[:batch_size], output_len=self.window_size)
        #h_l = torch.reshape(h_l, (batch_size * num_timesteps, -1))
        #h_g = h_g.view(batch_size * num_timesteps, -1)
        #angle = F.cosine_similarity(h_l, h_g, dim=1).pow(2).view(batch_size, num_timesteps).sum(1)
        angle = torch.zeros((1, ), device=x.device)
        return p_x_hat, local_dist, global_dist, z_t, z_g, angle.detach()

    def elbo(self, x):
        batch_size, num_timesteps, input_size = x.size()

        x_hat_dist, pz_t, p_zg, z_t, z_g, angle = self.forward(x)
        pz_mu = torch.cat((torch.zeros((batch_size, 1, self.local_size), device=x.device), pz_t.mean[:, :-1]), dim=1)
        pz_sd = torch.cat((torch.ones((batch_size, 1, self.local_size), device=x.device), pz_t.stddev[:, :-1]), dim=1)
        pz = D.Normal(pz_mu, pz_sd)
        cf_loss = torch.zeros((1, ), device=x.device)
        #if self.lamda!=0:
        #    z_g_2 = torch.randn(z_g.size(), device=x.device)
        #    cf_dist = self.decoder(z_t, z_g_2, output_len=self.window_size)
        #    _, pos_zg = self.global_encoder(cf_dist.rsample())
        #    cf_loss = (pos_zg.log_prob(z_g)-pos_zg.log_prob(z_g_2)).exp().mean(-1)
        p_zg_1 = D.Normal(p_zg.mean[:batch_size], p_zg.stddev[:batch_size])
        #p_zg_2 = D.Normal(p_zg.mean[batch_size:], p_zg.stddev[batch_size:])
        #cf_loss = p_zg_1.log_prob(z_g[batch_size:]) + p_zg_2.log_prob(z_g[:batch_size])

        nll = -x_hat_dist.log_prob(x)  # shape=(M*batch_size, time, dimensions)
        kl_l = D.kl.kl_divergence(pz_t, pz) / (x.size(1) // self.window_size)
        kl_l = kl_l.mean(dim=(1,2))  # shape=(M*batch_size, time, dimensions)

        nll = nll.mean(dim=(1,2))
        kl_g = D.kl.kl_divergence(p_zg_1, D.Normal(loc=0, scale=1.)).sum(-1)
        elbo = (-nll - self.beta * (kl_l + kl_g) - self.lamda * cf_loss).mean()  # shape=(M*batch_size)

        mse = F.mse_loss(x_hat_dist.mean, x).detach()

        return -elbo, nll.mean(), kl_l.mean(), kl_g.mean(), cf_loss.mean(), mse.mean()

    def training_step(self, batch, batch_ix):
        x, y = batch
        opt = self.optimizers()
        # Forward pass
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x)
        # Optimization
        opt.zero_grad()
        self.manual_backward(elbo)
        opt.step()
        #print(elbo, r)
        self.log_dict({"tr_elbo": elbo, "tr_nll": nll, "tr_kl_l": kl_l, "tr_kl_g": kl_g, "tr_mse": mse, "tr_cf": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x)
        self.log_dict({"va_elbo": elbo, "va_nll": nll, "va_kl_l": kl_l, "va_kl_g": kl_g, "va_mse": mse, "va_cf": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_elbo" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_elbo"])


class GLR(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Decoupled Global and Local Representation learning (GLR) model

        Attributes:
            global_encoder: Encoder model that learns the global representation for each time series sample
            local_encoder: Encoder model that learns the local representation of time series windows over time
            decoder: Decoder model that generated the time series sample distribution
            time_length: Maximum length of the time series samples
            data_dim: Input data dimension (number of features)
            window_size: Length of the time series window to learn representations for
            kernel: Gaussian Process kernels for different dimensions of local representations
            beta: KL divergence weight in loss term
            lamda: Counterfactual regularization weight in the loss term
            M: Number of Monte-Carlo samples
            lambda: Counterfactual regularization weight
            length_scale: Kernel length scale
            kernel_scales: number of different length scales over latent space dimensions
        """
        super().__init__(*args, **kwargs)
