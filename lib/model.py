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
                      ContmEncoderGlobal, TransformerEncoder,
                      ContEncoderGlobal, convEncoder, convDecoder)
from sklearn.linear_model import LinearRegression
from numbers import Number



class BaseModel(pl.LightningModule):
    def __init__(self, input_size, local_size, global_size, num_timesteps, window_size=20,
                 beta=1., gamma=1., mask_windows=0, lr=0.001, seed=42, local_dropout=0.0):
        super().__init__()
        self.window_step = 1
        self.num_timesteps = num_timesteps
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
        print("ELBO")
        print(x.shape)
        #(9000, 8, 64, 64, 3)
        batch_size = x.size()[0]
        num_timesteps = x.size()[1]
        input_size = np.prod(x.size()[2:])
        # batch_size, num_timesteps, input_size = x.size()
        #mask = mask.view(batch_size, num_timesteps, -1)

        x_hat_dist, pz_t, p_zg, z_t, z_g = self.forward(x, mask, window_step=self.window_step)
        if self.gamma!=0:
            cf_loss = self.calc_cf_loss(x, x_hat_dist, pz_t, p_zg, z_t, z_g)
        else:
            cf_loss = torch.zeros((1, ), device=x.device)

        print("BEFORE PERMUTE -1 ")
        print(x.shape)

        # x_w is size: (batch_size, num_windows, input_size, window_size)
        x_w = x.unfold(1, self.window_size, self.window_step)
        # Permute to: (batch_size, num_windows, window_size, input_size)
        print("BEFORE PERMUTE")
        print(x_w.shape)
        x_w = x_w.permute(0, 1, 3, 2)
        num_windows = x_w.size(1)
        x_w = torch.reshape(x_w, (batch_size, -1, input_size))
        nll = -x_hat_dist.log_prob(x_w)  # shape=(M*batch_size, time, dimensions)
        # Prior is previous timestep -> smoothness
        pz = D.Normal(pz_t.mean[:, :-1], pz_t.stddev[:, :-1])
        pz_t = D.Normal(pz_t.mean[:, 1:], pz_t.stddev[:, 1:])
        kl_l = D.kl.kl_divergence(pz_t, pz) #/ (x.size(1) // self.window_size)
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
            elbo = (-nll - self.anneal * self.beta * (kl_l + 0.0 * kl_g) 
                    - self.anneal * self.gamma * cf_loss).mean()  # shape=(M*batch_size)
        else:
            elbo = (-nll - self.beta * (kl_l + 0.0 * kl_g) 
                    - self.gamma * cf_loss).mean()  # shape=(M*batch_size)

        #measured_ratio = mask.float().mean(dim=(1, 2))
        #kl_l = kl_l * measured_ratio

        mse = F.mse_loss(x_hat_dist.mean, x_w).detach()
        
        return -elbo, nll.mean(), kl_l.mean(), kl_g.mean(), cf_loss.mean(), mse.mean()

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        raise NotImplementedError

    def training_step(self, batch, batch_ix):
        print("TRAINING???")
        x, mask, y = batch
        x = x.view(x.shape[0], x.shape[1], -1)
        opt = self.optimizers()
        print("FIRST??")
        print(x.shape)
        # Forward pass
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x, mask)
        # Optimization
        opt.zero_grad()
        self.manual_backward(elbo)
        #nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/500)
        self.log_dict({"tr_elbo_step": elbo, "tr_nll_step": nll, "tr_kl_l_step": kl_l, "tr_kl_g_step": kl_g, "tr_mse_step": mse, "tr_cf_step": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        print("VALIDATION")
        x, mask, y = batch
        x = x.view(x.shape[0], x.shape[1], -1)
        print(x.shape)
        #torch.Size([8, 8, 64, 64, 3])
        print("NEXT")
        elbo, nll, kl_l, kl_g, cf, mse = self.elbo(x, mask)
        self.log_dict({"va_elbo": elbo, "va_nll": nll, "va_kl_l": kl_l, "va_kl_g": kl_g, "va_mse": mse, "va_cf": cf}, prog_bar=True, on_epoch=True,
                        logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.90, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_elbo" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_elbo"])


class GLR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = EncoderGlobal(self.input_size, self.global_size, 32, 3)
        self.conv_encoder = convEncoder(128, 3)
        self.conv_decoder = convDecoder(32 + 256, 3)
    def forward(self, x, mask, window_step=None):
        batch_size = x.size()[0]
        num_timesteps = x.size()[1]
        print("FIRST SHAPE")
        print(x.shape)
        # torch.Size([8, 8, 64, 64, 3])
        #conv_x = self.encoder_frame(x)
        #print("SHAPE AFTER CONV ENCODE")
        #print(conv_x.shape)
        h_l, global_dist = self.global_encoder(x)
        h_g, local_dist = self.local_encoder(x, window_step=window_step)

        z_t = self.dropout(local_dist.rsample())
        z_g = global_dist.rsample()
        x_hat_mean = self.decoder(z_t, z_g[:batch_size], output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        # p_x_hat = self.conv_decoder(p_x_hat)
        return p_x_hat, local_dist, global_dist, z_t, z_g

    def calc_cf_loss(self, x, x_hat_dist, pz_t, p_zg, z_t, z_g):
        z_g_2 = torch.randn(z_g.size(), device=x.device)
        cf_mean = self.decoder(z_t, z_g_2, output_len=self.window_size)
        cf_dist = D.Normal(cf_mean, 0.1)
        _, pos_zg = self.global_encoder(cf_dist.rsample())
        cf_loss = (pos_zg.log_prob(z_g)-pos_zg.log_prob(z_g_2)).exp().mean(-1)
        return cf_loss

    def encoder_frame(self, x): 
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        # (9000, 8, 64, 64, 3)

        x = x.view(-1, 3, 64, 64)
        print("SHAPE BEFORE CONV ENCODE")
        print(x.shape)
        x_embed = self.conv_encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], 8, -1) 


class GLR_Original(BaseModel):
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

class ContGLR(GLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = ContEncoderGlobal(self.input_size, self.global_size, 32, 3, self.window_size, 0.3)

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
        batch_size = x.size()[0]
        num_timesteps = x.size()[1]
        h_g, local_dist = self.local_encoder(x, window_size=self.window_size)
        z_t = local_dist.rsample()
        x_hat_mean = self.decoder(z_t, None, output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, None, z_t, None

class TransGLR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_encoder = TransformerEncoder(self.input_size, self.global_size, 32, 6, self.num_timesteps)
        
    def forward(self, x, mask, window_step=None):
        batch_size = x.size()[0]
        num_timesteps = x.size()[1]
        h_l, global_dist = self.global_encoder(x)
        h_g, local_dist = self.local_encoder(x, window_step=window_step)
        z_t = self.dropout(local_dist.rsample())
        z_g = global_dist.rsample()
        x_hat_mean = self.decoder(z_t, z_g[:batch_size], output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, global_dist, z_t, z_g

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

class GlobalGLR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_mlp = nn.Sequential(
            nn.Linear(223, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        
        
    def forward(self, x, mask, window_step=None):
        batch_size = x.size()[0]
        num_timesteps = x.size()[1]
        #global_embeddings = F.normalize(self.global_embeddings, p=2, dim=-1)[mask.squeeze(1)]
        u = F.one_hot(mask, num_classes=223).float().squeeze(1)
        global_embeddings = self.u_mlp(u)
        #global_embeddings = F.normalize(global_embeddings, dim=-1)
        print(global_embeddings.size(), global_embeddings)
        #global_embeddings = self.global_embeddings / self.global_embeddings.max(0)[0]
        #global_embeddings = global_embeddings[mask.squeeze(1)]
        #global_embeddings = self.global_embeddings / torch.norm(self.global_embeddings, p=2, dim=-1).max()
        #print(torch.norm(global_embeddings, p=2, dim=-1))
        h_g, local_dist = self.local_encoder(x, window_step=window_step)
        z_t = self.dropout(local_dist.rsample())
        #z_g = global_dist.rsample()
        x_hat_mean = self.decoder(z_t, global_embeddings, output_len=self.window_size)
        p_x_hat = D.Normal(x_hat_mean, 0.1)
        return p_x_hat, local_dist, D.Normal(global_embeddings, 0.01), z_t, global_embeddings
