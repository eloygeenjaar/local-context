import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch import optim, nn
from torch import distributions as D
from torch.nn import functional as F
from scipy.stats import pearsonr
# Import all kernel functions
from .modules import (EncoderLocal,
                      WindowDecoder, EncoderGlobal,
                      TransformerEncoder)
from .conv import EncoderLocalConv, WindowDecoderConv, GlobalConv, LayeredConv
from .lin import (EncoderLocalLin, WindowDecoderLin,
                  EncoderLocalFull, WindowDecoderFull)
from .rnn import (
    EncoderLocalRNN, WindowDecoderRNN, mEncoderGlobal
)
from sklearn.linear_model import LinearRegression
from numbers import Number
from .utils import mask_input, mask_input_extra



class BaseModel(pl.LightningModule):
    def __init__(self, input_size, local_size, global_size, num_timesteps, window_size=20,
                 beta=1., gamma=1., mask_windows=0, lr=0.001, seed=42, local_dropout=0.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dropout = nn.Dropout(local_dropout)
        self.local_size = local_size
        self.global_size = global_size
        self.input_size = input_size
        self.window_size = window_size
        self.mask_windows = mask_windows
        self.lr = lr
        self.seed = seed
        self.transition_mat = nn.Linear(self.local_size, self.local_size)
        self.beta = beta
        self.gamma = gamma
        self.dropout_rate = 0.0
        self.anneal = 0.
        # Visible values
        self.mask_v = D.Bernoulli(0.95)
        # Interpolated values
        self.mask_i = D.Bernoulli(0.95)
        # Lightning parameters
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x, mask, window_step=None):
        raise NotImplementedError

    def elbo(self, x, x_full, validation=False):
        model_output = self.forward(x, x_full)
        mse = F.mse_loss(model_output['x_hat'], x, reduction='none').mean(dim=(0, 2, 3))
        kl_l = D.kl.kl_divergence(model_output['p_zt'], D.Normal(0., 1.)).mean(dim=(1, 2))
        if model_output['p_zg'] is not None:
            kl_g = D.kl.kl_divergence(p_zg, D.Normal(0., 1.)).mean(dim=(1))
        else:
            kl_g = torch.zeros((1, ), device=x.device)
        if self.training:
            loss = (mse + self.anneal * self.beta * (kl_l + 0.1 * kl_g)).mean(0)
        else:
            loss = (mse + self.beta * (kl_l + 0.1 * kl_g)).mean(0)
        
        #loss = mse
        return (loss, mse.detach().mean(), torch.zeros((1, )), torch.zeros((1, )),
                kl_l.detach().mean(), kl_g.detach().mean())

    def training_step(self, batch, batch_ix):
        x, task_mask, mask, y = batch
        #x_masked = mask_input_extra(x, task_mask)
        x_masked = x.unfold(1, 32, 2).permute(0, 1, 3, 2)
        x_masked = x_masked.float()
        opt = self.optimizers()
        # Forward passWindowDecoderFull
        loss, mse, mse_swap_t, mse_swap_s, kl_l, kl_g = self.elbo(x_masked, x.float())
        # Optimization
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.anneal = min(1.0, self.anneal + 1/5000)
        self.log_dict({"tr_loss": loss, "tr_mse": mse, "tr_kll": kl_l, "tr_klg": kl_g, "tr_swap_t": mse_swap_t, "tr_swap_s": mse_swap_s}, prog_bar=True, on_epoch=True,
                        logger=True)

    def validation_step(self, batch, batch_idx):
        x, task_mask, mask, y = batch
        x_masked = x.unfold(1, 32, 2).permute(0, 1, 3, 2)
        x_masked = x_masked.float()
        opt = self.optimizers()
        # Forward passWindowDecoderFull
        loss, mse, mse_swap_t, mse_swap_s, kl_l, kl_g = self.elbo(x_masked, x.float())
        self.log_dict({"va_loss": loss, "va_mse": mse, "va_kll": kl_l, "va_klg": kl_g, "va_swap_t": mse_swap_t, "va_swap_s": mse_swap_s}, prog_bar=True, on_epoch=True,
                        logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.90, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if "va_loss" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["va_loss"])

class LR(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.local_encoder = EncoderLocalConv(self.input_size, self.local_size, self.window_size, 128, 3)
        #self.decoder = WindowDecoderConv(self.input_size, self.local_size, 0, self.window_size, 128, 3)
        self.local_encoder = EncoderLocalRNN(self.input_size, self.local_size, self.window_size, 128, 3)
        self.decoder = WindowDecoderRNN(self.input_size, self.local_size, 0, self.window_size, 128, 3)
        
    def forward(self, x, mask, window_step=None):
        local_dist = self.local_encoder(x, mask=mask)
        z_t = local_dist.rsample()
        x_hat = self.decoder(z_t, None, output_len=self.window_size)
        return {
            'x_hat': x_hat,
            'x_hat_swap_t': None,
            'x_hat_swap_s': None,
            'p_zt': local_dist,
            'p_zg': None,
            'l2_t': torch.zeros((1,), device=x.device),
            'l2_g': torch.zeros((1,), device=x.device)}

class LRConv(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalConv(self.input_size, self.local_size, self.window_size, 128, 3)
        self.decoder = WindowDecoderConv(self.input_size, self.local_size, 0, self.window_size, 128, 3)
        
    def forward(self, x, mask, window_step=None):
        local_dist = self.local_encoder(x, mask=mask)
        z_t = local_dist.rsample()
        x_hat = self.decoder(z_t, None, output_len=self.window_size)
        return {
            'x_hat': x_hat,
            'x_hat_swap_t': None,
            'x_hat_swap_s': None,
            'p_zt': local_dist,
            'p_zg': None,
            'l2_t': torch.zeros((1,), device=x.device),
            'l2_g': torch.zeros((1,), device=x.device)}

class LRLin(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalLin(self.input_size, self.local_size, self.window_size, 128, 3)
        self.decoder = WindowDecoderLin(self.input_size, self.local_size, self.global_size, self.window_size, 128, 3)
        
    def forward(self, x, mask, window_step=None):
        local_dist = self.local_encoder(x, mask=mask)
        z_t = local_dist.rsample()
        x_hat = self.decoder(z_t, output_len=self.window_size)
        return x_hat, local_dist

class LRFull(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalFull(self.input_size, self.local_size, self.window_size, 128, 3)
        self.decoder = WindowDecoderFull(self.input_size, self.local_size, self.global_size, self.window_size, 128, 3)
        
    def forward(self, x, mask, window_step=None):
        local_dist = self.local_encoder(x, mask=mask)
        z_t = local_dist.rsample()
        x_hat = self.decoder(z_t, output_len=self.window_size)
        return x_hat, local_dist

class LRRNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalRNN(self.input_size, self.local_size, self.window_size, 128, 3)
        self.decoder = WindowDecoderRNN(self.input_size, self.local_size, self.global_size, self.window_size, 128, 3)
        
    def forward(self, x, mask, window_step=None):
        local_dist = self.local_encoder(x)
        z_t = local_dist.rsample()
        x_hat = self.decoder(z_t, None, output_len=self.window_size)
        return x_hat, local_dist, None


class GlobalLRRNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalRNN(self.input_size, self.local_size, self.window_size, 128, 3)
        self.global_encoder = TransformerEncoder(self.input_size, self.global_size, 64, 8, self.num_timesteps)
        self.decoder = WindowDecoderRNN(self.input_size, self.local_size, self.global_size, self.window_size, 128, 3)
        
    def forward(self, x, x_full, window_step=None):
        local_dist = self.local_encoder(x)
        global_dist = self.global_encoder(x_full)
        z_t = local_dist.rsample()
        z_g = global_dist.rsample()
        x_hat = self.decoder(z_t, z_g, output_len=self.window_size)
        return x_hat, local_dist, global_dist

class SwapGlobalLRRNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_encoder = EncoderLocalRNN(self.input_size, self.local_size, self.window_size, 128, 3)
        #self.global_encoder = TransformerEncoder(self.input_size, self.global_size, 64, 8, self.num_timesteps)
        self.global_encoder = mEncoderGlobal(self.input_size, self.global_size, 256, 3)
        self.decoder = WindowDecoderRNN(self.input_size, self.local_size, self.global_size, self.window_size, 128, 3)
        self.shared = self.local_size // 2
        
    def forward(self, x, x_full, window_step=None):
        #batch_size = x.size(1)
        local_dist = self.local_encoder(x)
        global_dist = self.global_encoder(x_full)
        z_t = local_dist.rsample()
        batch_size, num_windows, local_size = z_t.size()
        z_g = global_dist.rsample()
        x_hat = self.decoder(z_t, z_g, output_len=self.window_size)
        # Swap current timestep with random other subjects (roll over batch)
        z_t_swap = z_t.clone()
        #z_t_swap = torch.roll(z_t, 1, 0).clone()
        #z_t_swap[..., :self.shared] = z_t_swap[torch.randperm(batch_size), :, :self.shared]
        z_t_swap = z_t_swap[torch.randperm(batch_size)]
        x_hat_swap_t = self.decoder(z_t_swap, z_g, output_len=self.window_size)
        l2_t = l2_g = torch.zeros((1, ), device=x.device)
        #l2_t = -0.05 * F.cosine_similarity(torch.reshape(z_t_swap, (-1, self.local_size)),
        #                                   torch.reshape(z_t, (-1, self.local_size)), dim=-1).mean()
        #l2_t = 0.1 * (z_t[:, 1:, self.shared:] - z_t[:, :-1, self.shared:]).abs().mean()
        return {
            'x_hat': x_hat,
            'x_hat_swap_t': x_hat_swap_t,
            'x_hat_swap_s': None,
            'p_zt': local_dist,
            'p_zg': global_dist,
            'l2_t': l2_t,
            'l2_g': torch.zeros((1, ), device=x.device)}

    def elbo(self, x, x_full, validation=False):
        model_output = self.forward(x, x_full)
        mse_swap_t = mse_swap_s = torch.zeros((1, ), device=x.device)
        mse_swap_t = F.mse_loss(model_output['x_hat_swap_t'], x, reduction='none').mean(dim=(0, 2, 3))
        #mse_swap_s = F.mse_loss(model_output['x_hat_swap_s'], x, reduction='none').mean(dim=(0, 2, 3))
        mse = F.mse_loss(model_output['x_hat'], x, reduction='none').mean(dim=(0, 2, 3))
        kl_l = D.kl.kl_divergence(model_output['p_zt'], D.Normal(0., 1.)).mean(dim=(1, 2))
        if model_output['p_zg'] is not None:
            #p_zg = D.Normal(model_output['p_zg'].mean.unsqueeze(1).repeat(1, x.size(2), 1),
            #                model_output['p_zg'].stddev.unsqueeze(1).repeat(1, x.size(2), 1))
            #kl_g = D.kl.kl_divergence(
            #    D.Normal(model_output['p_zt'].mean[..., self.shared:],
            #             model_output['p_zt'].stddev[..., self.shared:]), p_zg).mean(dim=(1, 2))
            kl_g = D.kl.kl_divergence(model_output['p_zg'], D.Normal(0., 1.)).mean(dim=-1)
        else:
            kl_g = torch.zeros((1, ), device=x.device)
        if self.training:
            loss = (mse + mse_swap_t + mse_swap_s + self.anneal * (self.beta * kl_l + self.gamma * kl_g + 
                    model_output['l2_t'] + model_output['l2_g'])).mean(0)
        else:
            loss = (mse + mse_swap_t + mse_swap_s + self.beta * kl_l + self.gamma * kl_g + 
                    model_output['l2_t'] + model_output['l2_g']).mean(0)
        
        #loss = mse
        return (loss, mse.detach().mean(), mse_swap_t.detach().mean(), mse_swap_s.detach().mean(),
                kl_l.detach().mean(), kl_g.detach().mean())

class GlobalVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_decoder = nn.Sequential(
            nn.Linear(self.local_size + self.global_size, 128, bias=True), nn.ELU(True), nn.Dropout(0.1),
            nn.Linear(128, 128, bias=True), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(128, self.input_size))
        self.encoder = LayeredConv(self.input_size, self.local_size, self.global_size)
        
    def forward(self, x, x_full, window_step=None):
        window_size, batch_size, num_windows, voxels = x.size()
        dists, z = self.encoder(x_full)
        x_hat = self.local_decoder(z)
        return {
            'x_hat': x_hat,
            'dists': dists}

    def elbo(self, x, x_full, validation=False):
        model_output = self.forward(x, x_full)
        mse_swap_t = mse_swap_s = torch.zeros((1, ), device=x.device)
        mse = F.mse_loss(model_output['x_hat'], x, reduction='none').mean(dim=(1, 2, 3))
        gamma = 1.0
        kl = gamma * D.kl.kl_divergence(model_output['dists'][0], D.Normal(0., 1.)).mean(dim=(1, 2))
        #print(f'0: {kl.detach().mean()}')
        for (i, dist) in enumerate(model_output['dists'][1:]):
            #print(f'{i+1}: {D.kl.kl_divergence(dist, D.Normal(0., 1.)).detach().mean()}')
            gamma *= 0.5
            kl += gamma * D.kl.kl_divergence(dist, D.Normal(0., 1.)).mean(dim=(1))
        kl /= len(model_output['dists'])
        if self.training:
            loss = (mse + mse_swap_t + mse_swap_s + self.anneal * (self.beta * kl)).mean(0)
        else:
            loss = (mse + mse_swap_t + mse_swap_s + self.beta * kl).mean(0)
        #loss = mse
        return (loss, mse.detach().mean(), mse_swap_t.detach().mean(), mse_swap_s.detach().mean(),
                kl.detach().mean(), torch.zeros((1, )))
