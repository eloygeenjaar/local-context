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
from .modules import TemporalEncoder, LocalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from numbers import Number
from sklearn.decomposition import PCA
from info_nce import InfoNCE



class BaseModel(pl.LightningModule):
    def __init__(self, input_size, local_size, global_size,
                 beta=1., gamma=1., lr=0.001, seed=42):
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.input_size = input_size
        self.spatial_decoder = nn.Sequential(
            nn.Linear(local_size + global_size, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, input_size),
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

    def elbo(self, x, x_p):
        output = self.forward(x, x_p)
        mse = F.mse_loss(output['x_hat'], x, reduction='none').mean(dim=(0, 2))
        kl_l = D.kl.kl_divergence(output['local_dist'], output['prior_dist']).mean(dim=(0, 2))
        if output['global_dist'] is not None:
            kl_g = D.kl.kl_divergence(output['global_dist'], D.Normal(0., 1.)).mean(dim=1)
        else:
            kl_g = torch.zeros((1, ), device=x.device)
        if output['global_dist_n'] is not None:
            cf_loss = self.cf_loss(output['global_dist'].mean, output['global_dist_p'].mean, output['global_dist_n'].mean)
        else:
            cf_loss = torch.zeros((1, ), device=x.device)
        loss = (mse + self.anneal * (self.beta * kl_l + self.gamma * cf_loss)).mean()
        return loss, {
            'mse': mse.detach().mean(),
            'kl_l': kl_l.detach().mean(),
            'kl_g': kl_g.detach().mean(),
            'cf': cf_loss.detach().mean()}, (output['local_dist'].mean.detach(), output['global_dist'].mean.detach())

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        raise NotImplementedError

    def training_step(self, batch, batch_ix):
        x, x_p, _, y = batch
        x = x.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        opt = self.optimizers()
        # Forward pass
        loss, output, _ = self.elbo(x, x_p)
        # Optimization
        opt.zero_grad()
        self.manual_backward(loss)
        #nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/5000)
        self.log_dict({"tr_loss": loss, "tr_mse": output['mse'],
                       "tr_kl_l": output['kl_l'], "tr_kl_g": output['kl_g'], "tr_cf": output['cf']}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, x_p, _, y = batch
        x = x.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        loss, output, (lo, gl) = self.elbo(x, x_p)
        lr = LogisticRegression()
        lo = lo.cpu().numpy()
        gl = gl.cpu().numpy()
        y = y.cpu().numpy()
        acc = lr.fit(gl, y).score(gl, y)
        self.log_dict({"va_loss": loss, "va_mse": output['mse'],
                       "va_kl_l": output['kl_l'], "va_kl_g": output['kl_g'],
                       "va_acc": acc, "va_cf": output['cf']}, prog_bar=True, on_epoch=True,
                        logger=True)
        if gl.shape[-1] > 2:
            pca = PCA(n_components=2)
            gl = pca.fit_transform(gl)
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(gl[y==0, 0], gl[y==0, 1], color='b', alpha=0.5)
        axs[0].scatter(gl[y==1, 0], gl[y==1, 1], color='r', alpha=0.5)
        axs[0].set_xlabel('Latent dimension 1')
        axs[0].set_ylabel('Latent dimension 2')
        axs[0].set_title('Global representations')
        axs[0].axis('equal')
        axs[1].scatter(lo[:, y==0, ..., 0].flatten(), lo[:, y==0, ..., 1].flatten(), color='b', alpha=0.2)
        axs[1].scatter(lo[:, y==1, ..., 0].flatten(), lo[:, y==1, ..., 1].flatten(), color='r', alpha=0.2)
        axs[1].set_xlabel('Latent dimension 1')
        axs[1].set_ylabel('Latent dimension 2')
        axs[1].set_title('Local representations')
        axs[1].axis('equal')
        plt.tight_layout()
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
        if "va_loss" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_loss"])


class DSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(self.input_size, self.local_size, self.global_size)
        
    def forward(self, x, x_p):
        # Reduce spatial dimension
        global_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': global_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'global_dist_n': None
        }

    def embed_global(self, x):
        return self.temporal_encoder.get_global(x)[0]

class LVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = LocalEncoder(self.input_size, 256, self.local_size)
        
    def forward(self, x, x_p):
        # Reduce spatial dimension
        local_dist, prior_dist = self.local_encoder(x)
        z = local_dist.rsample()
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': D.Normal(local_dist.mean.mean(0), 1.),
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'global_dist_n': None
        }

class CDSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(self.input_size, self.local_size, self.global_size)
        self.cf_loss = InfoNCE()
        self.tau = 1.
        
    def forward(self, x, x_p):
        # Reduce spatial dimension
        global_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        x_n = x[:, torch.randperm(x.size(1))]
        global_dist_n, _ = self.temporal_encoder.get_global(x_n)
        global_dist_p, _ = self.temporal_encoder.get_global(x_p)
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': global_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'global_dist_n': global_dist_n,
            'global_dist_p': global_dist_p
        }

    def embed_global(self, x):
        return self.temporal_encoder.get_global(x)[0]
    
class SCDSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(self.input_size, self.local_size, self.global_size)
        self.cf_loss = InfoNCE()
        self.tau = 1.
        
    def forward(self, x, x_p):
        # Reduce spatial dimension
        global_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        x_n = x[:, torch.randperm(x.size(1))]
        global_dist_n, _ = self.temporal_encoder.get_global(x_n)
        global_dist_p, _ = self.temporal_encoder.get_global(x_p)
        x_hat = self.spatial_decoder(z)
        return {
            'global_dist': global_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'global_dist_n': global_dist_n,
            'global_dist_p': global_dist_p
        }

    def embed_global(self, x):
        return self.temporal_encoder.get_global(x)[0]
    
    def elbo(self, x, x_p):
        output = self.forward(x, x_p)
        mse = F.mse_loss(output['x_hat'], x, reduction='none').mean(dim=(0, 2))
        kl_l = D.kl.kl_divergence(output['local_dist'], output['prior_dist']).mean(dim=(0, 2))
        if output['global_dist'] is not None:
            kl_g = D.kl.kl_divergence(output['global_dist'], D.Normal(0., 1.)).mean(dim=1)
        else:
            kl_g = torch.zeros((1, ), device=x.device)
        if output['global_dist_n'] is not None:
            cf_loss = self.cf_loss(output['global_dist'].mean[..., :2], output['global_dist_p'].mean[..., :2], output['global_dist_n'].mean[..., :2])
        else:
            cf_loss = torch.zeros((1, ), device=x.device)
        loss = (mse + self.anneal * (self.beta * kl_l + self.gamma * cf_loss)).mean()
        return loss, {
            'mse': mse.detach().mean(),
            'kl_l': kl_l.detach().mean(),
            'kl_g': kl_g.detach().mean(),
            'cf': cf_loss.detach().mean()}, (output['local_dist'].mean.detach(), output['global_dist'].mean.detach())
