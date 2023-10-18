import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import distributions as D


class EncoderLocal(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderLocal, self).__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                          bidirectional=True)
        # Create encoder_net
        hidden_sizes = [(2 * hidden_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.local_size)
        self.lv_layer = nn.Linear(hidden_size, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        zl_mean, zl_std, h_ts = [], [], []
        if window_step is None:
            window_step = self.window_size
        # x_w is size: (batch_size, num_windows, input_size, window_size)
        x_w = x.unfold(1, self.window_size, window_step)
        # Permute to: (batch_size, num_windows, window_size, input_size)
        x_w = x_w.permute(0, 1, 3, 2)
        for t in range(x_w.size(1)):
            h_t, h_T = self.gru(x_w[:, t])
            h_T = torch.reshape(h_T.permute(1, 0, 2), (x.size(0), 2 * self.hidden_size))
            features = self.mlp(h_T)
            zl_mean.append(self.mu_layer(features))
            zl_std.append(F.softplus(self.lv_layer(features)))
            h_ts.append(h_t)
        zl_mean = torch.stack(zl_mean, dim=1)
        zl_std = torch.stack(zl_std, dim=1)
        h_t = torch.cat(h_ts, dim=1)
        return h_t, D.Normal(zl_mean, zl_std)

class EncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=2 * hidden_size, batch_first=True)
        hidden_sizes = [(2 * hidden_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        h_t, h_T = self.gru(x)
        features = self.mlp(h_T.squeeze(0))
        zg_mean = self.mu_layer(features)
        zg_std = F.softplus(self.lv_layer(features))
        return h_t, D.Normal(zg_mean, zg_std)

class mEncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(mEncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        hidden_sizes = [(2 * hidden_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        h_t, h_T = self.gru(x)
        features = self.mlp(h_t).mean(1)
        zg_mean = self.mu_layer(features)
        zg_std = F.softplus(self.lv_layer(features))
        return h_t, D.Normal(zg_mean, zg_std)

class ContmEncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers, window_size, perc_mask):
        """Initializes the instance"""
        super(ContmEncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        hidden_sizes = [(2 * hidden_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)
        self.perc_mask = perc_mask
        self.window_size = window_size

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        batch_size, num_timesteps, _ = x.size()
        num_windows = num_timesteps // self.window_size
        num_masked = int(self.perc_mask * num_windows)
        x_mask = x.clone()
        for i in range(batch_size):
            window_ixs = torch.randperm(num_windows, device=x.device)[:num_masked]
            for ix in window_ixs:
                x_mask[i, (self.window_size * ix):(self.window_size * (ix + 1))] = 0.
        x = torch.cat((x, x_mask))
        h_t, h_T = self.gru(x)
        features = self.mlp(h_t).mean(1)
        zg_mean = self.mu_layer(features)
        zg_std = F.softplus(self.lv_layer(features))
        return h_t, D.Normal(zg_mean, zg_std)

class WindowDecoder(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(WindowDecoder, self).__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.output_size = output_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        # This had a BatchNorm
        self.embedding = nn.Sequential(*[
            #nn.LayerNorm(local_size + global_size),
            #nn.Linear(local_size + global_size, 2 * (local_size + global_size)),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(2 * (local_size + global_size), local_size + global_size),
            #nn.LeakyReLU(inplace=True),
            nn.Linear(local_size + global_size, 8 * hidden_size)
            ])
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size * 8, batch_first=True)
        hidden_sizes = [(local_size + global_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.factor_layer = nn.Linear(8 * hidden_size, local_size + global_size)
        self.mu_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, prior_len, _ = z_t.size()
        if z_g is not None:
            z = torch.cat((z_t, z_g.unsqueeze(1).repeat(1, z_t.size(1), 1)), dim=-1)
        else:
            z = z_t
        emb = self.embedding(z)
        recon_seq = []
        # For each window ...
        in_x = torch.zeros((batch_size, output_len, 1), device=z_t.device)
        for t in range(z_t.size(1)):
            rnn_out, _ = self.gru(in_x, emb[:, t].unsqueeze(0).contiguous())
            recon_seq.append(rnn_out)
        # Stitch windows together
        recon_seq = torch.cat(recon_seq, dim=1)
        with torch.no_grad():
            self.factor_layer.weight.data = F.normalize(self.factor_layer.weight.data, dim=1)
        recon_seq = self.factor_layer(recon_seq)
        recon_seq = self.mlp(recon_seq)
        x_mean = self.mu_layer(recon_seq)
        return x_mean
