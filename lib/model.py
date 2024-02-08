import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ray import train
from torch import nn, optim
from info_nce import InfoNCE
import lightning.pytorch as pl
from torch import distributions as D
from torch.nn import functional as F
from sklearn.decomposition import PCA
from .modules import (
    TemporalEncoder, LocalEncoder, LocalDecoder,
    ConvContextEncoder, ConvContextDecoder)
from sklearn.linear_model import LogisticRegression


class BaseModel(pl.LightningModule):
    def __init__(self, num_layers, spatial_hidden_size, temporal_hidden_size, lr, batch_size, beta, gamma, theta):
        super().__init__()
        self.local_size = config['local_size']
        self.context_size = config['context_size']
        self.input_size = config['data_size']
        self.window_size = config['window_size']
        self.contrastive_dim = config['contrastive_dim']
        self.num_layers = hyperparameters['num_layers']
        self.spatial_hidden_size = hyperparameters['spatial_hidden_size']
        self.temporal_hidden_size = hyperparameters['temporal_hidden_size']
        self.dropout = hyperparameters['dropout']
        self.spatial_decoder = LocalDecoder(
            config['local_size'] + config['context_size'],
            config['spatial_hidden_size'],
            config['data_size'], hyperparameters['num_layers'],
            dropout_val=hyperparameters['dropout'])
        self.lr = hyperparameters['lr']
        self.seed = config['seed']
        # Loss function hyperparameters
<<<<<<< HEAD
        self.beta = hyperparameters['beta']
        self.gamma = hyperparameters['gamma']
        self.theta = hyperparameters['theta']
        self.lambda_ = hyperparameters['lambda']
=======
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
>>>>>>> ray_classification
        self.anneal = 0.0
        self.loss_keys = [
            'mse',
            'kl_l',
            'kl_c',
            'cont',
            'cf',
            'elbo'
        ]
        # Lightning parameters
        self.automatic_optimization = False

    def forward(self, x, x_p):
        raise NotImplementedError

    def elbo(self, x, x_p):
        # Output is a dictionary
        output = self.forward(x, x_p)
        upsampling = x.size(0) // self.window_size
        if upsampling > 1:
            x = x[(upsampling // 2)::upsampling]
        # Calculate reconstruction loss
        mse = -D.Normal(output['x_hat'], 0.1).log_prob(x).mean(dim=(0, 2))
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
            smooth_loss = torch.zeros((1, ), device=x.device)
        # Calculate the contrastive loss (in case we are training
        # a contrastive model)
        if output['context_dist_n'] is not None:
            # Default: contrastive_dim = context_dim
            # but we can also try to contrast contexts only along a certain
            # dimension to 'disentangle' similar and dissimilar dimensions
            # in the context space.
            cont_loss = self.cf_loss(
                output['context_dist'].mean[..., :self.contrastive_dim],
                output['context_dist_p'].mean[..., :self.contrastive_dim],
                output['context_dist_n'].mean[..., :self.contrastive_dim])
        else:
            cont_loss = torch.zeros((1, ), device=x.device)
        # Calculate the counterfactual loss (in case we are training
        # a counterfactual model)
        if output['cf_local_z'] is not None:
            #log_prob = output['local_dist'].log_prob(output['local_z'])
            #cf_log_prob = output['local_dist'].log_prob(output['cf_local_z'])
            # Calculate likelihood:
            # p(z | x, z_c) / p(z* | x, z_c*)
            cf_loss = -F.cosine_similarity(output['nf_local_z'],
                                           output['cf_local_z'], dim=-1).mean(0)
        else:
            cf_loss = torch.zeros((1, ), device=x.device)
        
        elbo = (mse + self.anneal * (
                self.beta * kl_l +
                self.gamma * kl_c
        ))
        # This is to make sure the scheduler optimizes theta
        # based on the elbo, not based on the cf_loss as well
        # since it will trivially lead to a very low theta.
        loss = (elbo + 
                self.anneal * self.theta * cont_loss +
                self.anneal * self.lambda_ * cf_loss).mean()
        return loss, {
            'mse': mse.detach().mean(),
            'kl_l': kl_l.detach().mean(),
            'kl_c': kl_c.detach().mean(),
            'cont': cont_loss.detach().mean(),
            'cf': cf_loss.detach().mean(),
            'elbo': elbo.detach().mean()
            }, (output['local_dist'].mean.detach(),
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
        #nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/5000)
        train_dict = {f'tr_{l_key}': output[l_key] for l_key in self.loss_keys}
        train_dict['tr_loss'] = loss.detach()
        self.log_dict(train_dict, on_step=False, prog_bar=False,
                      on_epoch=True, logger=False, sync_dist=True)

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
        self.log_dict(va_dict, on_step=False, prog_bar=False,
                      on_epoch=True, logger=False, sync_dist=True)
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
            optimizer, factor=0.90, patience=10, verbose=True)
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
            hidden_size=self.temporal_hidden_size, dropout_val=self.dropout)

    def forward(self, x, x_p):
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        upsampling = x.size(0) // self.window_size
        if upsampling > 1:
            z = z[(upsampling // 2)::upsampling]
            local_dist = D.Normal(
                local_dist.mean[(upsampling // 2)::upsampling],
                local_dist.stddev[(upsampling // 2)::upsampling]
            )
            prior_dist = D.Normal(
                prior_dist.mean[(upsampling // 2)::upsampling],
                prior_dist.stddev[(upsampling // 2)::upsampling]
            )
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

    def generate_counterfactual(self, x, cf_context):
        # The input to this function should essentiall look like this:
        # x: (num_timesteps, batch_size, data_size)
        # cf_context: (batch_size, context_size)
        # Each subject in the batch should be a subject we want to 
        # generate a counterfactual for, where
        # each entry in the cf_context batch should correspond
        # to the counterfactual context we should use for the 
        # corresponding subject in x.
        # Context: (2, batch_size, context_size)
        # Local: (num_timesteps, 2, batch_size, local_size)
        # The 2 dimensions are: 0: non-cf, 1: cf
        local, z = self.temporal_encoder.generate_counterfactual(x, cf_context)
        x_hat = self.spatial_decoder(z)
        return local, x_hat


class CFDSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            hidden_size=self.temporal_hidden_size, dropout_val=self.dropout)

    def forward(self, x, x_p):
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        # Concatenated as follows: (local_z, context_z)
        # Global z is the repeated across the timestep dimension so
        # we just select the first
        global_z = z[0, ..., self.local_size:]
        #context_distances = torch.cdist(global_z, global_z)
        #ix = torch.argmax(context_distances, dim=1)
        # Generate counterfactual local representations based on
        # the furthest context window.
        ix = torch.randperm(x.size(1))
        cf_local_dist, _ = self.temporal_encoder.generate_counterfactual(
            x, global_z[ix])
        cf_local_z = cf_local_dist.rsample()
        nf_local_z = z[..., :self.local_size].clone()
        upsampling = x.size(0) // self.window_size
        if upsampling > 1:
            z = z[(upsampling // 2)::upsampling]
            local_dist = D.Normal(
                local_dist.mean[(upsampling // 2)::upsampling],
                local_dist.stddev[(upsampling // 2)::upsampling]
            )
            prior_dist = D.Normal(
                prior_dist.mean[(upsampling // 2)::upsampling],
                prior_dist.stddev[(upsampling // 2)::upsampling]
            )
        x_hat = self.spatial_decoder(z)
        return {
            'context_dist': context_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            # Concatenated as follows: (local_z, context_z)
            'nf_local_z': nf_local_z,
            'cf_local_z': cf_local_z,
            # This is only used for contrastive models
            'context_dist_n': None
        }

    def embed_context(self, x):
        return self.temporal_encoder.get_context(x)[0]

    def generate_counterfactual(self, x, cf_context):
        # The input to this function should essentiall look like this:
        # x: (num_timesteps, batch_size, data_size)
        # cf_context: (batch_size, context_size)
        # Each subject in the batch should be a subject we want to 
        # generate a counterfactual for, where
        # each entry in the cf_context batch should correspond
        # to the counterfactual context we should use for the 
        # corresponding subject in x.
        # Context: (2, batch_size, context_size)
        # Local: (num_timesteps, 2, batch_size, local_size)
        # The 2 dimensions are: 0: non-cf, 1: cf
        local, z = self.temporal_encoder.generate_counterfactual(x, cf_context)
        x_hat = self.spatial_decoder(z)
        return local, x_hat


class LVAE(BaseModel):
    # Local VAE (only has a local encoder, not a local and context encoder)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = LocalEncoder(
            self.input_size, self.temporal_hidden_size, self.local_size,
            dropout_val=self.dropout)

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
            hidden_size=self.temporal_hidden_size, dropout_val=self.dropout)
        self.cf_loss = InfoNCE()
        self.tau = 1.

    def forward(self, x, x_p):
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x)
        upsampling = x.size(0) // self.window_size
        if upsampling > 1:
            z = z[(upsampling // 2)::upsampling]
            local_dist = D.Normal(
                local_dist.mean[(upsampling // 2)::upsampling],
                local_dist.stddev[(upsampling // 2)::upsampling]
            )
            prior_dist = D.Normal(
                prior_dist.mean[(upsampling // 2)::upsampling],
                prior_dist.stddev[(upsampling // 2)::upsampling]
            )
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

    def generate_counterfactual(self, x, cf_context):
        # The input to this function should essentiall look like this:
        # x: (num_timesteps, batch_size, data_size)
        # cf_context: (batch_size, context_size)
        # Each subject in the batch should be a subject we want to 
        # generate a counterfactual for, where
        # each entry in the cf_context batch should correspond
        # to the counterfactual context we should use for the 
        # corresponding subject in x.
        # Context_z: (num_timesteps, 2, batch_size, context_size)
        # Context: (2, batch_size, context_size)
        # Local: (num_timesteps, 2, batch_size, local_size)
        # The 2 dimensions are: 0: non-cf, 1: cf
        num_timesteps, batch_size, data_size = x.size()
        local, z = self.temporal_encoder.generate_counterfactual(x, cf_context)
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.view(num_timesteps, 2, batch_size, data_size)
        return local, x_hat


class IDSVAE(DSVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            hidden_size=self.temporal_hidden_size, independence=True,
            dropout_val=self.dropout)

class CIDSVAE(CDSVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            hidden_size=self.temporal_hidden_size, independence=True,
            dropout_val=self.dropout)

class CO(BaseModel):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs)
        #()
        # {'num_layers': 1, 'spatial_hidden_size': 128, 'temporal_hidden_size': 128, 'lr': 0.0010872422452394674, 'batch_size': 128, 'beta': 0, 'gamma': 0.000291063591313307, 'theta': 0}

        super().__init__(*args, **kwargs)
        self.temporal_encoder = ConvContextEncoder(
            self.input_size, self.spatial_hidden_size, self.context_size,
            self.window_size
        )
        self.spatial_decoder = ConvContextDecoder(
            self.context_size, self.spatial_hidden_size, self.input_size,
            self.window_size
        )

    def forward(self, x, x_p):
        x = x.permute(1, 2, 0)
        context_dist = self.temporal_encoder(x)
        z = context_dist.rsample()
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.permute(2, 0, 1)
        return {
            'context_dist': context_dist,
            'local_dist': D.Normal(torch.zeros_like(x_hat), 1.),
            'prior_dist': D.Normal(torch.zeros_like(x_hat), 1.),
            'x_hat': x_hat,
            'context_dist_n': None,
            'context_dist_p': None
        }

    def embed_context(self, x):
        return self.temporal_encoder(x)[0]
