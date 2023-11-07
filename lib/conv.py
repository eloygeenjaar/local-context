import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D


class EncoderLocalConv(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderLocalConv, self).__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.layers = nn.ModuleList([
            nn.Conv1d(53, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.25)
        ])
        self.mu_layer = nn.Linear(4 * 128, self.local_size)
        self.lv_layer = nn.Linear(4 * 128, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        zl_mean, zl_std, h_ts = [], [], []
        if window_step is None:
            window_step = self.window_size
        batch_size, input_size, window_size = x.size()
        #x = x.clone() * mask
        x = torch.reshape(x, (batch_size, input_size, window_size))
        for layer in self.layers:
            x = layer(x)
        x = x.view(batch_size, -1)
        zl_mean = self.mu_layer(x)
        zl_std = F.softplus(self.lv_layer(x))
        return D.Normal(zl_mean, 0.01)


class WindowDecoderConv(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(WindowDecoderConv, self).__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.output_size = output_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        # This had a BatchNorm
        self.embedding = nn.Sequential(*[
            nn.Linear(local_size + global_size, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25)
            ])
        self.layers = nn.ModuleList([
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 53, kernel_size=3, stride=2, padding=1, bias=True),
        ])
        self.output_sizes = [(8, ), (16, ), (32, ), (30, )]

    def forward(self, z_t, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, local_size = z_t.size()
        z = z_t
        emb = self.embedding(z)
        emb = torch.reshape(emb, (batch_size, 128, 4))
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.ConvTranspose1d):
                emb = layer(emb, output_size=self.output_sizes[i])
                i += 1
            else:
                emb = layer(emb)
        emb = emb.view(batch_size, -1, output_len)
        return emb
