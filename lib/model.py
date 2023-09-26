import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch import optim, nn
from torch import distributions as D
from torch.nn import functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from .modules import MLP
from .utils import mean_corr_coef as mcc
from numbers import Number


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))

def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))

# From https://github.com/ilkhem/iVAE/blob/master/models/nets.py
class BaseModel(pl.LightningModule):
    def __init__(self, input_size, latent_dim, aux_dim, size_dataset,
                 num_layers=3, activation='xtanh', hidden_dim=50, slope=.1,
                 hyperparam_tuple=(1., 1., 1., 1.)):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.size_dataset = size_dataset
        self.num_layers = num_layers
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.slope = slope
        self.a, self.b, self.c, self.d = hyperparam_tuple

        # prior params
        self.prior_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        # decoder params
        self.f = MLP(latent_dim, input_size, hidden_dim, num_layers, activation=activation, slope=slope)
        #self.decoder_sd = nn.Parameter((.1 * torch.ones(1)).sqrt(), requires_grad=False)
        self.decoder_var = nn.Parameter(.1 * torch.ones(1), requires_grad=False)
        # encoder params
        self.automatic_optimization = False
        self.save_hyperparameters()

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        raise NotImplementedError

    def decoder(self, z):
        rec = self.f(z)
        return rec

    def prior(self, u):
        raise NotImplementedError

    def forward(self, x, u):
        prior_mu, prior_var = self.prior(u)
        mu, var = self.encoder(x, u)
        z = self.reparameterize(mu, var)
        rec = self.decoder(z)
        return rec, mu, var, z, prior_var

    def elbo(self, x, gt, u):
        f, g, v, z, l = self.forward(x, u)
        batch_size, num_timesteps, input_size = x.size()

        x = x.view(batch_size * num_timesteps, input_size)
        u = u.view(batch_size * num_timesteps, self.aux_dim)
        gt = gt.view(batch_size * num_timesteps, self.latent_dim)
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, batch_size, d)
        logqs_tmp = log_normal(
            z.view(batch_size * num_timesteps, 1, self.latent_dim),
            g.view(1, batch_size * num_timesteps, self.latent_dim),
            v.view(1, batch_size * num_timesteps, self.latent_dim))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(batch_size * num_timesteps * self.size_dataset)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False)
                   - np.log(batch_size * num_timesteps * self.size_dataset)).sum(dim=-1)

        elbo = -(self.a * logpx -
                 self.b * (logqs_cux - logqs) -
                 self.c * (logqs - logqs_i) -
                 self.d * (logqs_i - logps_cu)).mean()
        r = mcc(gt, z)
        return elbo, r

    def training_step(self, batch, batch_ix):
        x, gt, u = batch
        opt = self.optimizers()
        # Forward pass
        elbo, r = self.elbo(x, gt, u)
        # Optimization
        opt.zero_grad()
        self.manual_backward(elbo)
        opt.step()
        #print(elbo, r)
        self.log_dict({"tr_elbo": elbo, "tr_corr": r}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, gt, u = batch
        elbo, r = self.elbo(x, gt, u)
        self.log_dict({"va_elbo": elbo, "va_corr": r}, prog_bar=True, on_epoch=True,
                        logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_elbo" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_elbo"])

class VAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = MLP(self.input_size, self.latent_dim, self.hidden_dim, self.num_layers,
                     activation=self.activation, slope=self.slope)
        self.logv = MLP(self.input_size, self.latent_dim, self.hidden_dim, self.num_layers,
                        activation=self.activation, slope=self.slope)
        self.prior_var = nn.Parameter(torch.ones(1), requires_grad=False)
        #self.apply(weights_init)

    def prior(self, u):
        return self.prior_mean, self.prior_var

    def encoder(self, x, u):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()


class iVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = MLP(self.input_size + self.aux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                     activation=self.activation, slope=self.slope)
        self.logv = MLP(self.input_size + self.aux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                        activation=self.activation, slope=self.slope)
        self.logl = MLP(self.aux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                        activation=self.activation, slope=self.slope)
        #self.apply(weights_init)

    def prior(self, u):
        batch_size, num_timesteps, aux_dim = u.size()
        u = u.view(batch_size * num_timesteps, aux_dim)
        logl = self.logl(u)
        return self.prior_mean, torch.exp(logl)

    def encoder(self, x, u):
        batch_size, num_timesteps, input_size = x.size()
        x = x.view(batch_size * num_timesteps, input_size)
        u = u.view(batch_size * num_timesteps, self.aux_dim)
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()


class HiVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faux_dim = 2
        self.g = MLP(self.input_size + self.faux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                     activation=self.activation, slope=self.slope)
        self.logv = MLP(self.input_size + self.faux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                        activation=self.activation, slope=self.slope)
        self.logl = MLP(self.faux_dim, self.latent_dim, self.hidden_dim, self.num_layers,
                        activation=self.activation, slope=self.slope)
        self.h_gru = nn.Parameter(torch.randn(1, 1, 50))
        self.z_gru = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.gru = nn.GRU(input_size=self.latent_dim, hidden_size=50, batch_first=True)
        self.gru_to_u = nn.Linear(50, self.faux_dim)
        #self.apply(weights_init)

    def forward(self, x, u):
        mu, var, z, prior_var, _ = self.encoder(x, u)
        rec = self.decoder(z)
        return rec, mu, var, z, prior_var

    def encoder(self, x, u):
        batch_size, num_timesteps, in_size = x.size()
        z_t = self.z_gru.repeat(batch_size, 1, 1)
        h_t = self.h_gru.repeat(1, batch_size, 1)
        mus, vars, zs, prior_vars, us = [], [], [], [], []
        for t in range(num_timesteps):
            _, h_t = self.gru(z_t, h_t)
            u_t = self.gru_to_u(h_t).squeeze(0)
            x_t = torch.cat((x[:, t], u_t), dim=-1)
            mu = self.g(x_t)
            var = self.logv(x_t).exp()
            z = self.reparameterize(mu, var)
            z_t = z.unsqueeze(1)
            prior_var = self.logl(u_t).exp()
            mus.append(mu)
            vars.append(var)
            zs.append(z)
            prior_vars.append(prior_var)
            us.append(u_t)
        mus = torch.stack(mus, dim=1).view(batch_size * num_timesteps, self.latent_dim)
        vars = torch.stack(vars, dim=1).view(batch_size * num_timesteps, self.latent_dim)
        z = torch.stack(zs, dim=1).view(batch_size * num_timesteps, self.latent_dim)
        prior_var = torch.stack(prior_vars, dim=1).view(batch_size * num_timesteps, self.latent_dim)
        u = torch.stack(us, dim=1)
        return mus, vars, z, prior_var, u
