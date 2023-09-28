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
        self.window_size = window_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # Create encoder_net
        lin_layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)] * num_layers
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.local_size)
        self.lv_layer = nn.Linear(hidden_size, self.local_size)

    def forward(self, x, mask=None, window_size=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        zl_mean, zl_std, h_ts = [], [], []
        for t in range(0, x.size(1) - window_size + 1, window_size):
            h_t, h_T = self.gru(x[:, t:t + window_size, :])
            features = self.mlp(h_T.squeeze(0))
            zl_mean.append(self.mu_layer(features))
            zl_std.append(F.softplus(self.lv_layer(features)))
            h_ts.append(h_t)
        zl_mean = torch.stack(zl_mean, dim=1)
        zl_std = torch.stack(zl_std, dim=1)
        h_t = torch.cat(h_ts, dim=1)
        return h_t, D.Normal(zl_mean, zl_std)

class EncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers, window_size, num_windows_mask):
        """Initializes the instance"""
        super(EncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        lin_layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)] * num_layers
        self.mlp = nn.Sequential(*lin_layers)
        self.window_size = window_size
        self.num_windows_mask = num_windows_mask
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)

    def forward(self, x, training=False):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        x_mask = x.clone()
        if training:
            x_mask = x_mask.repeat(2, 1, 1)
            num_windows = x.size(1) // self.window_size
            window_ixs = torch.randint(low=0, high=(num_windows + 1), size=(x_mask.size(0), self.num_windows_mask, ),
                                    device=x.device)
            for i in range(window_ixs.size(0)):
                for j in range(self.num_windows_mask):
                    x_mask[i, window_ixs[i, j]:(window_ixs[i, j] + self.window_size)] = 0.
        h_t, h_T = self.gru(x_mask)
        features = self.mlp(h_T.squeeze(0))
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
            nn.Linear(local_size + global_size, hidden_size),
            nn.Tanh()])
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        lin_layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)] * num_layers
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, prior_len, _ = z_t.size()
        z = torch.cat((z_t, z_g.unsqueeze(1).repeat(1, z_t.size(1), 1)), dim=-1)
        emb = self.embedding(z)
        recon_seq = []
        # For each window ...
        in_x = torch.zeros((batch_size, output_len, 1), device=z_t.device)
        for t in range(z_t.size(1)):
            rnn_out, _ = self.gru(in_x, emb[:, t].unsqueeze(0).contiguous())
            recon_seq.append(rnn_out)
        # Stitch windows together
        recon_seq = torch.cat(recon_seq, dim=1)
        recon_seq = self.mlp(recon_seq)
        x_mean = self.mu_layer(recon_seq)
        return D.Normal(x_mean, 0.1)
