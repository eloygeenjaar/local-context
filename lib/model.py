import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch import optim, nn
from torch import distributions as D
from torch.nn import functional as F
from scipy.stats import pearsonr
# Import all kernel functions
from .modules import (EncoderLocal, mEncoderGlobal,
                      WindowDecoder, EncoderGlobal,
                      ContmEncoderGlobal)
from sklearn.linear_model import LinearRegression
from numbers import Number


class BaseModel(pl.LightningModule):
    def __init__(self, input_size, local_size, global_size, window_size=20,
                 beta=1., gamma=1., mask_windows=0, lr=0.001, seed=42, local_dropout=0.0):
        super().__init__()
        self.window_step = 1
        self.dropout = nn.Dropout(local_dropout)
        self.local_size = local_size
        self.global_size = global_size
        self.local_encoder = EncoderLocal(input_size, local_size, window_size, 32, 3)
        self.input_size = input_size
        self.window_size = window_size
        self.mask_windows = mask_windows
        self.lr = lr
        self.seed = seed
        self.decoder = WindowDecoder(input_size, local_size, global_size, window_size, 32, 4)
        self.beta = beta
        self.gamma = gamma
        self.anneal = 0.0
        # Lightning parameters
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x, mask, window_step=None):
        raise NotImplementedError

    def elbo(self, x, mask):
        batch_size, num_timesteps, input_size = x.size()
        mask = mask.view(batch_size, num_timesteps, -1)

        x_hat_dist, pz_t, p_zg, z_t, z_g = self.forward(x, mask, window_step=self.window_step)
        if self.gamma!=0:
            cf_loss = self.calc_cf_loss(x, x_hat_dist, pz_t, p_zg, z_t, z_g)
        else:
            cf_loss = torch.zeros((1, ), device=x.device)

        # x_w is size: (batch_size, num_windows, input_size, window_size)
        x_w = x.unfold(1, self.window_size, self.window_step)
        # Permute to: (batch_size, num_windows, window_size, input_size)
        x_w = x_w.permute(0, 1, 3, 2)
        num_windows = x_w.size(1)
        x_w = torch.reshape(x_w, (batch_size, -1, input_size))
        nll = -x_hat_dist.log_prob(x_w)  # shape=(M*batch_size, time, dimensions)
        # Prior is previous timestep -> smoothness
        pz_mu = torch.cat((torch.zeros((batch_size, 1, self.local_size), device=x.device), pz_t.mean[:, :-1]), dim=1)
        pz_sd = torch.cat((torch.ones((batch_size, 1, self.local_size), device=x.device), pz_t.stddev[:, :-1]), dim=1)
        pz = D.Normal(pz_mu, pz_sd)
        kl_l = D.kl.kl_divergence(pz_t, D.Normal(0, 1.)) #/ (x.size(1) // self.window_size)
        #kl_l = kl_l.sum(1).mean(-1)  # shape=(M*batch_size, time, dimensions)
        kl_l = kl_l.mean(dim=(1, 2))
        # I use the inverse of the original mask
        #nll = torch.where(mask==0, torch.zeros_like(nll), nll)

        nll = nll.mean(dim=(1, 2))
        if p_zg is not None:
            p_zg = D.Normal(p_zg.mean[:batch_size], p_zg.stddev[:batch_size])
            kl_g = D.kl.kl_divergence(p_zg, D.Normal(loc=0, scale=1.)).sum(-1)
        else:
            kl_g = torch.zeros((1,), device=x.device)
        if self.training:
            elbo = (-nll - self.anneal * self.beta * (kl_l + kl_g) 
                    - self.anneal * self.gamma * cf_loss).mean()  # shape=(M*batch_size)
        else:
            elbo = (-nll - self.beta * (kl_l + kl_g) 
                    - self.gamma * cf_loss).mean()  # shape=(M*batch_size)

        #measured_ratio = mask.float().mean(dim=(1, 2))
        #kl_l = kl_l * measured_ratio

        mse = F.mse_loss(x_hat_dist.mean, x_w).detach()
        
        return -elbo, nll.mean(), kl_l.mean(), kl_g.mean(), cf_loss.mean(), mse.mean()

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        raise NotImplementedError

    def training_step(self, batch, batch_ix):
        x, mask, y = batch
        opt = self.optimizers()
        # Forward pass
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x, mask)
        # Optimization
        opt.zero_grad()
        self.manual_backward(elbo)
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/500)
        self.log_dict({"tr_elbo": elbo, "tr_nll": nll, "tr_kl_l": kl_l, "tr_kl_g": kl_g, "tr_mse": mse, "tr_cf": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, mask, y = batch
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x, mask)
        self.log_dict({"va_elbo": elbo, "va_nll": nll, "va_kl_l": kl_l, "va_kl_g": kl_g, "va_mse": mse, "va_cf": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_elbo" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_elbo"])


class GLR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = EncoderGlobal(self.input_size, self.global_size, 32, 3)
        
    def forward(self, x, mask, window_step=None):
        batch_size, num_timesteps, _ = x.size()
        h_l, global_dist = self.global_encoder(x)
        h_g, local_dist = self.local_encoder(x, window_step=window_step)
        z_t = self.dropout(local_dist.rsample())
        z_g = global_dist.rsample()
        x_hat_mean = self.decoder(z_t, z_g[:batch_size], output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, global_dist, z_t, z_g

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        z_g_2 = torch.randn(z_g.size(), device=x.device)
        cf_mean = self.decoder(z_t, z_g_2, output_len=self.window_size)
        cf_dist = D.Normal(cf_mean, 0.1)
        _, pos_zg = self.global_encoder(cf_dist.rsample())
        cf_loss = (pos_zg.log_prob(z_g)-pos_zg.log_prob(z_g_2)).exp().mean(-1)
        return cf_loss

class mGLR(GLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = mEncoderGlobal(self.input_size, self.global_size, 32, 3)

class ContmGLR(GLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = ContmEncoderGlobal(self.input_size, self.global_size, 32, 3, self.window_size, 0.3)

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        batch_size = x.size(0)
        anchor = D.Normal(p_zg.mean[:batch_size], p_zg.stddev[:batch_size])
        pos = D.Normal(p_zg.mean[batch_size:], p_zg.stddev[batch_size:])
        anchor_ix, neg_ix = torch.triu_indices(batch_size, batch_size, device=x.device)
        anchor_neg = D.Normal(p_zg.mean[anchor_ix], p_zg.stddev[anchor_ix])
        neg = D.Normal(p_zg.mean[neg_ix], p_zg.stddev[neg_ix])
        return self.triplet_loss(anchor, pos, anchor_neg, neg)
    
    @staticmethod
    def wasserstein_dist(dist_a, dist_b):
        m_dist = (dist_a.mean - dist_b.mean).pow(2).sum(-1)
        s_dist = torch.norm(dist_a.stddev - dist_b.stddev, dim=-1).pow(2)
        return m_dist + s_dist
    
    def triplet_loss(self, anchor_pos, pos, anchor_neg, neg, margin=1.0):
        return torch.max(
            self.wasserstein_dist(anchor_pos, pos).mean(0) -
            self.wasserstein_dist(anchor_neg, neg).mean(0) + margin,
            torch.zeros((1, ), device=pos.mean.device))[0]

class VAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocal(
            self.input_size,
            self.local_size + self.global_size,
            self.window_size, 64, 3)
        self.local_size = self.local_size + self.global_size
    
    def forward(self, x, mask):
        batch_size, num_timesteps, _ = x.size()
        h_g, local_dist = self.local_encoder(x, window_size=self.window_size)
        z_t = local_dist.rsample()
        x_hat_mean = self.decoder(z_t, None, output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, None, z_t, None

class LinVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_encoder = nn.utils.parametrizations.weight_norm(nn.Linear(64 * 64 * 64, self.input_size))
        self.lin_decoder = nn.utils.parametrizations.weight_norm(nn.Linear(self.input_size, 64 * 64 * 64))
    
    def forward(self, x, mask):
        batch_size, num_timesteps, _ = x.size()
        x = self.lin_encoder(x)
        h_g, local_dist = self.local_encoder(x, window_size=self.window_size)
        z_t = local_dist.rsample()
        x_hat_mean = self.decoder(z_t, None, output_len=self.window_size)
        x_hat_mean = self.lin_decoder(x_hat_mean)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, None, z_t, None

class LinGLR(GLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_encoder = nn.utils.parametrizations.weight_norm(nn.Linear(64 * 64 * 64, self.input_size))
        self.lin_decoder = nn.utils.parametrizations.weight_norm(nn.Linear(self.input_size, 64 * 64 * 64))

    def forward(self, x, mask):
        batch_size, num_timesteps, _ = x.size()
        x = self.lin_encoder(x)
        h_l, global_dist = self.global_encoder(x)
        h_g, local_dist = self.local_encoder(x, window_size=self.window_size)
        z_t = local_dist.rsample()
        z_g = global_dist.rsample()
        x_hat_mean = self.decoder(z_t, z_g, output_len=self.window_size)
        x_hat_mean = self.lin_decoder(x_hat_mean)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, global_dist, z_t, z_g

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        z_g_2 = torch.randn(z_g.size(), device=x.device)
        cf_mean = self.decoder(z_t, z_g_2, output_len=self.window_size)
        cf_dist = D.Normal(cf_mean, 0.1)
        _, pos_zg = self.global_encoder(cf_dist.rsample())
        cf_loss = (pos_zg.log_prob(z_g)-pos_zg.log_prob(z_g_2)).exp().mean(-1)
        return cf_loss

class LinmGLR(LinGLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = mEncoderGlobal(self.input_size, self.global_size, 64, 3)
