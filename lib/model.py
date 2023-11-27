import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch import optim, nn
from torch import distributions as D
from torch.nn import functional as F
from scipy.stats import pearsonr
# Import all kernel functions
from .modules import TemporalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from numbers import Number



class BaseModel(pl.LightningModule):
    def __init__(self, input_size, local_size, global_size,
                 beta=1., gamma=1., lr=0.001, seed=42):
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.input_size = input_size
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_size, 32)
        )
        self.spatial_decoder = nn.Sequential(
            #nn.Linear(local_size + global_size, 32), nn.LeakyReLU(True),
            #nn.Linear(32, input_size)
            nn.Linear(local_size + global_size, input_size)
        )
        self.lr = lr
        self.seed = seed
        self.beta = beta
        self.gamma = gamma
        self.anneal = 0.0
        # Lightning parameters
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x, mask, window_step=None):
        raise NotImplementedError

    def elbo(self, x):
        output = self.forward(x)
        mse = F.mse_loss(output['x_hat'], x, reduction='none').mean(dim=(0, 2))
        #kl_l = D.kl.kl_divergence(output['local_dist'], output['prior_dist']).mean(dim=(0, 2))
        kl_l = D.kl.kl_divergence(output['local_dist'], D.Normal(0., 1.)).mean(dim=(0, 2))
        kl_g = D.kl.kl_divergence(output['global_dist'], D.Normal(0., 1.)).mean(dim=1)
        loss = (mse + self.anneal * (self.beta * kl_l + self.gamma * kl_g)).mean()
        return loss, {
            'mse': mse.detach().mean(),
            'kl_l': kl_l.detach().mean(),
            'kl_g': kl_g.detach().mean()}, (output['local_dist'].mean.detach(), output['global_dist'].mean.detach())

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        raise NotImplementedError

    def training_step(self, batch, batch_ix):
        x, y = batch
        x = x.permute(1, 0, 2)
        opt = self.optimizers()
        # Forward pass
        loss, output, _ = self.elbo(x)
        # Optimization
        opt.zero_grad()
        self.manual_backward(loss)
        #nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/500)
        self.log_dict({"tr_loss": loss, "tr_mse": output['mse'],
                       "tr_kl_l": output['kl_l'], "tr_kl_g": output['kl_g']}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.permute(1, 0, 2)
        loss, output, (lo, gl) = self.elbo(x)
        lr = LogisticRegression()
        lo = lo.cpu().numpy()
        gl = gl.cpu().numpy()
        y = y.cpu().numpy()
        acc = lr.fit(gl, y).score(gl, y)
        self.log_dict({"va_loss": loss, "va_mse": output['mse'],
                       "va_kl_l": output['kl_l'], "va_kl_g": output['kl_g'],
                       "va_acc": acc}, prog_bar=True, on_epoch=True,
                        logger=True)
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(gl[y==0, 0], gl[y==0, 1], color='b', alpha=0.5)
        axs[0].scatter(gl[y==1, 0], gl[y==1, 1], color='r', alpha=0.5)
        axs[0].set_xlabel('Latent dimension 1')
        axs[0].set_ylabel('Latent dimension 2')
        axs[0].set_title('Global representations')
        axs[0].axis('equal')
        axs[1].scatter(lo[:, y==0, ..., 0].flatten(), lo[:, y==0, ..., 1].flatten(), color='b', alpha=0.5)
        axs[1].scatter(lo[:, y==1, ..., 0].flatten(), lo[:, y==1, ..., 1].flatten(), color='r', alpha=0.5)
        axs[1].set_xlabel('Latent dimension 1')
        axs[1].set_ylabel('Latent dimension 2')
        axs[1].set_title('Local representations')
        axs[1].axis('equal')
        plt.savefig(f'{self.logger.save_dir}/global.png', dpi=200)
        #plt.savefig(f'{self.logger.save_dir}/{self.logger.version}/global.png', dpi=200)
        plt.clf()
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.90, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_elbo" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_elbo"])


class DSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(32, self.local_size, self.global_size)
        
    def forward(self, x):
        # Reduce spatial dimension
        x = self.spatial_encoder(x)
        global_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': global_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat
        }

class VAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_layer = nn.Linear(32, self.local_size)
        self.lv_layer = nn.Linear(32, self.local_size)
        
    def forward(self, x):
        # Reduce spatial dimension
        x = self.spatial_encoder(x)
        mu = self.mu_layer(x)
        sd = torch.exp(0.5 * self.lv_layer(x))
        local_dist = D.Normal(mu, sd)
        z = local_dist.rsample()
        global_dist = D.Normal(torch.zeros((x.size(1), 2), device=x.device),
                               torch.ones((x.size(1), 2), device=x.device))
        prior_dist = D.Normal(0., 1.)
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': global_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat
        }