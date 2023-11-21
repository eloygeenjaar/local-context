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
            nn.Conv1d(input_size, 32, kernel_size=3, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1)
        ])
        self.mu_layer = nn.Linear(4 * 128, self.local_size)
        self.lv_layer = nn.Linear(4 * 128, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        window_size, batch_size, num_windows, voxels = x.size()
        x = x.permute(1, 2, 3, 0).view(batch_size * num_windows, self.input_size, window_size)
        for layer in self.layers:
            x = layer(x)
        x = x.view(batch_size, num_windows, -1)
        zl_mean = self.mu_layer(x)
        zl_std = F.softplus(self.lv_layer(x))
        return D.Normal(zl_mean, zl_std)


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
            nn.Linear(local_size, 512, bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
            ])
        self.layers = nn.ModuleList([
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, self.output_size, kernel_size=3, stride=2, padding=1, bias=True),
        ])
        self.output_sizes = [(8, ), (16, ), (32, ), (30, )]

    def forward(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, num_windows, local_size = z_t.size()
        z = z_t
        emb = self.embedding(z)
        emb = torch.reshape(emb, (batch_size * num_windows, 128, 4))
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.ConvTranspose1d):
                emb = layer(emb, output_size=self.output_sizes[i])
                i += 1
            else:
                emb = layer(emb)
        emb = emb.view(batch_size, num_windows, self.output_size, output_len)
        emb = emb.permute(3, 0, 1, 2)
        return emb

class GlobalConv(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super().__init__()
        self.input_size = input_size
        self.global_size = global_size
        self.layers = nn.ModuleList([
            nn.Conv1d(input_size, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2, stride=2, padding=1, dilation=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=2, dilation=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=4, dilation=8, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=2, stride=2, padding=8, dilation=16, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=2, stride=2, padding=16, dilation=32, bias=False),
            nn.BatchNorm1d(1024),
            nn.AvgPool1d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1)
        ])
        self.mu_layer = nn.Linear(1024, self.global_size)
        self.lv_layer = nn.Linear(1024, self.global_size)

    def forward(self, x):
        batch_size, timesteps, voxels = x.size()
        ix = torch.randint(timesteps-1024, device=x.device, size=(1, ))
        x_new = x[:, ix:(ix + 1024)].permute(0, 2, 1)
        for layer in self.layers:
            x_new = layer(x_new)
        mu = self.mu_layer(x_new)
        sd = torch.exp(0.5 * self.lv_layer(x_new))
        return D.Normal(mu, sd)

class ConvBlock(nn.Module):
    def __init__(self, size, out):
        super().__init__()
        self.size = size
        self.conv = nn.Sequential(
            nn.Conv1d(size, out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out, out, kernel_size=3, stride=1, padding=4, dilation=4, bias=True),
            nn.ELU(inplace=True),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.conv(x)


class LayeredConv(nn.Module):
    def __init__(self, input_size: int, local_size: int, global_size: int):
        super().__init__()
        self.input_size = input_size
        self.mlp = nn.Sequential(nn.Linear(input_size, 128), nn.ELU(inplace=True), nn.Dropout(0.1),
                                 nn.Linear(128, 128), nn.ELU(inplace=True), nn.Dropout(0.1))
        self.lin = nn.Sequential(nn.Linear(128, 64), nn.ELU(inplace=True))
        self.mu_layer_lin = nn.Linear(128, local_size)
        self.lv_layer_lin = nn.Linear(128, local_size)

        self.hidden_sizes = [64, 64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList(
            [ConvBlock(size, out) for (size, out) in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])]
        )
        self.windowed_avgpool = nn.AvgPool1d(32, stride=2)
        self.avgpools = nn.ModuleList(
            [nn.AvgPool1d(size, size) for size in [4, 4, 4, 4, 4]]
        )

        self.mu_layer = nn.Linear(1024, global_size)
        self.lv_layer = nn.Linear(1024, global_size)

    def forward(self, x):
        dists, zs = [], []
        batch_size, timesteps, voxels = x.size()
        x = self.mlp(x)
        mu = self.mu_layer_lin(x)
        sd = torch.exp(0.5 * self.lv_layer_lin(x)).clamp(1E-5, 5)
        dist = D.Normal(mu, sd)
        z = dist.rsample()
        z = z.unfold(1, 32, 2).permute(0, 1, 3, 2)
        dists.append(dist)
        zs.append(z)
        x = self.lin(x)
        # (batch_size, voxels, timesteps)
        x = x.permute(0, 2, 1)
        for (layer, avgpool) in zip(self.layers, self.avgpools):
            # (batch_size, channels, num_timesteps // size)
            # (batch_size, channels, timesteps)
            x = layer(x)
            x = avgpool(x)
            # (batch_size, timesteps, channels)
            #if x.size(-1) != 1024:
            #    x_w = torch.repeat_interleave(x, repeats=1024//x.size(-1), dim=-1)
            #else:
            #    x_w = x.clone()
            #x_w = self.windowed_avgpool(x_w)
            #x_w = x_w.permute(0, 2, 1)
        x = x.squeeze(-1)
        mu = self.mu_layer(x)
        sd = torch.exp(0.5 * self.lv_layer(x)).clamp(1E-5, 5)
        dist = D.Normal(mu, sd)
        z = dist.rsample().unsqueeze(1).unsqueeze(-2).repeat(1, z.size(1), z.size(2), 1)
        dists.append(dist)
        zs.append(z)

        z = torch.cat(zs, dim=-1)
        return dists, z