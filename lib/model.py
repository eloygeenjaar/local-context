import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ray import train
from torch import optim
from info_nce import InfoNCE
import lightning.pytorch as pl
from torch import distributions as D
from torch.nn import functional as F
from sklearn.decomposition import PCA
from .modules import (
    TemporalEncoder, LocalEncoder, LocalDecoder)
from sklearn.linear_model import LogisticRegression


class BaseModel(pl.LightningModule):
    def __init__(self, config, hyperparameters, viz):
        super().__init__()
        self.local_size = config['local_size']
        self.context_size = config['context_size']
        self.input_size = config['data_size']
        self.num_layers = hyperparameters['num_layers']
        self.spatial_hidden_size = hyperparameters['spatial_hidden_size']
        self.temporal_hidden_size = hyperparameters['temporal_hidden_size']
        self.spatial_decoder = LocalDecoder(
            config['local_size'] + config['context_size'],
            config['spatial_hidden_size'],
            config['data_size'], hyperparameters['num_layers'])
        self.lr = hyperparameters['lr']
        self.seed = config['seed']
        # Loss function hyperparameters
        self.beta = hyperparameters['beta']
        self.gamma = hyperparameters['gamma']
        self.theta = hyperparameters['theta']
        self.anneal = 0.0
        self.viz = viz
        self.loss_keys = [
            'mse',
            'kl_l',
            'kl_c',
            'cf'
        ]
        # Lightning parameters
        self.automatic_optimization = False
        self.save_hyperparameters(hyperparameters)

    def forward(self, x, x_p):
        raise NotImplementedError

    def elbo(self, x, x_p):
        # Output is a dictionary
        output = self.forward(x, x_p)
        # Calculate reconstruction loss
        mse = F.mse_loss(output['x_hat'], x, reduction='none').mean(dim=(0, 2))
        # Calculate the kl-divergence for the local representations
        kl_l = D.kl.kl_divergence(
            output['local_dist'],
            output['prior_dist']).mean(dim=(0, 2))
        # Calculate the context kl-divergence
        if output['context_dist'] is not None:
            kl_c = D.kl.kl_divergence(
                output['context_dist'], D.Normal(0., 1.)).mean(dim=1)
        else:
            kl_c = torch.zeros((1, ), device=x.device)
        # Calculate the contrastive loss (in case we are training
        # a contrastive model)
        if output['context_dist_n'] is not None:
            # Default: contrastive_dim = context_dim
            # but we can also try to contrast contexts only along a certain
            # dimension to 'disentangle' similar and dissimilar dimensions
            # in the context space.
            cf_loss = self.cf_loss(
                output['context_dist'].mean[..., :self.contrastive_dim],
                output['context_dist_p'].mean[..., :self.contrastive_dim],
                output['context_dist_n'].mean[..., :self.contrastive_dim])
        else:
            cf_loss = torch.zeros((1, ), device=x.device)
        loss = (mse + self.anneal * (
            self.beta * kl_l +
            self.gamma * kl_c +
            self.theta * cf_loss)).mean()
        return loss, {
            'mse': mse.detach().mean(),
            'kl_l': kl_l.detach().mean(),
            'kl_c': kl_c.detach().mean(),
            'cf': cf_loss.detach().mean()}, (
                output['local_dist'].mean.detach(),
                output['context_dist'].mean.detach())

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        raise NotImplementedError

    def training_step(self, batch, batch_ix):
        x, x_p = batch[0:2]
        x = x.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        opt = self.optimizers()
        # Forward pass
        loss, output, _ = self.elbo(x, x_p)
        # Optimization
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/5000)
        train_dict = {f'tr_{l_key}': output[l_key] for l_key in self.loss_keys}
        train_dict['tr_loss'] = loss.detach()
        self.log_dict(train_dict, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, x_p, _, y = batch
        x = x.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        loss, output, (local_z, context_z) = self.elbo(x, x_p)
        # Calculate validation accuracy
        lr = LogisticRegression()
        local_z = local_z.cpu().numpy()
        context_z = context_z.cpu().numpy()
        y = y.cpu().numpy()
        acc = lr.fit(context_z, y).score(context_z, y)
        va_dict = {f'va_{l_key}': output[l_key] for l_key in self.loss_keys}
        va_dict['va_acc'] = acc
        va_dict['va_loss'] = loss.detach()
        self.log_dict(va_dict, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        if self.viz:# Visualize representations
            if context_z.shape[-1] > 2:
                pca = PCA(n_components=2)
                context_z = pca.fit_transform(context_z)
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(context_z[y == 0, 0], context_z[y == 0, 1],
                        color='b', alpha=0.5)
            axs[0].scatter(context_z[y == 1, 0], context_z[y == 1, 1],
                        color='r', alpha=0.5)
            axs[0].set_xlabel('Latent dimension 1')
            axs[0].set_ylabel('Latent dimension 2')
            axs[0].set_title('Context representations')
            axs[0].axis('equal')
            axs[1].scatter(local_z[:, y == 0, ..., 0].flatten(),
                        local_z[:, y == 0, ..., 1].flatten(),
                        color='b', alpha=0.2)
            axs[1].scatter(local_z[:, y == 1, ..., 0].flatten(),
                        local_z[:, y == 1, ..., 1].flatten(),
                        color='r', alpha=0.2)
            axs[1].set_xlabel('Latent dimension 1')
            axs[1].set_ylabel('Latent dimension 2')
            axs[1].set_title('Local representations')
            axs[1].axis('equal')
            plt.tight_layout()
            plt.savefig(f'{self.logger.save_dir}/context.png', dpi=200)
            # TODO": save these files in the specific lightning_logs dir
            # (low priority)
            # plt.savefig(
            # f'{self.logger.save_dir}/{self.logger.version}/context.png', dpi=200)
            plt.clf()
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.90, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_loss" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_loss"])


class DSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            self.temporal_hidden_size)

    def forward(self, x, x_p):
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        x_hat = self.spatial_decoder(z)
        return {
            'context_dist': context_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            # This is only used for contrastive models
            'context_dist_n': None
        }

    def embed_context(self, x):
        return self.temporal_encoder.get_context(x)[0]


class LVAE(BaseModel):
    # Local VAE (only has a local encoder, not a local and context encoder)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = LocalEncoder(
            self.input_size, self.temporal_hidden_size, self.local_size)

    def forward(self, x, x_p):
        local_dist, prior_dist = self.local_encoder(x)
        z = local_dist.rsample()
        x_hat = self.spatial_decoder(z)
        return {
            'context_dist': D.Normal(local_dist.mean.mean(0), 1.),
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'context_dist_n': None
        }


class CDSVAE(BaseModel):
    # Contrastive DSVAE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            self.temporal_hidden_size)
        self.cf_loss = InfoNCE()
        self.tau = 1.

    def forward(self, x, x_p):
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        # Create negative example
        # (another window in the batch, which is randomized)
        x_n = x[:, torch.randperm(x.size(1))]
        # Generate a context representation for the negative sample
        context_dist_n, _ = self.temporal_encoder.get_context(x_n)
        # Generate a context representation for the positive sample
        context_dist_p, _ = self.temporal_encoder.get_context(x_p)
        x_hat = self.spatial_decoder(z)
        return {
            'context_dist': context_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'context_dist_n': context_dist_n,
            'context_dist_p': context_dist_p
        }

    def embed_context(self, x):
        return self.temporal_encoder.get_context(x)[0]
