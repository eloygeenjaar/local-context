import torch
from torch import nn
from torch import distributions as D


class EncoderLocalLin(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.layers = nn.Sequential(
            nn.Linear(input_size * window_size, hidden_size),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_size, self.local_size)
        #self.lv_layer = nn.Linear(256, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        batch_size, input_size, window_size = x.size()
        zl_mean = self.mu_layer(self.layers(x.view(batch_size, -1)))
        return D.Normal(zl_mean, 0.01)


class WindowDecoderLin(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.output_size = output_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        # This had a BatchNorm
        self.layers = nn.Sequential(
            nn.Linear(local_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_size, (output_size * window_size))

    def forward(self, z_t, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, local_size = z_t.size()
        emb = self.mu_layer(self.layers(z_t))
        emb = emb.view(batch_size, -1, self.window_size)
        return emb

class EncoderLocalFull(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )
        self.mu_layer = nn.Linear(hidden_size, self.local_size)
        #self.lv_layer = nn.Linear(256, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        batch_size, input_size, window_size = x.size()
        x = x.permute(0, 2, 1)
        zl_mean = self.mu_layer(self.layers(x))
        return D.Normal(zl_mean, 0.01)


class WindowDecoderFull(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super().__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.output_size = output_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        # This had a BatchNorm
        self.layers = nn.Sequential(
            nn.Linear(local_size, hidden_size),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z_t, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, window_size, local_size = z_t.size()
        emb = self.mu_layer(self.layers(z_t))
        emb = emb.permute(0, 2, 1)
        emb = emb.view(batch_size, -1, self.window_size)
        return emb
