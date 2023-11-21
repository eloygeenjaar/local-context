import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D


class EncoderLocalRNN(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderLocalRNN, self).__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.lin = nn.utils.parametrizations.weight_norm(nn.Linear(input_size, local_size))
        self.gru = nn.GRU(input_size=local_size, hidden_size=128, batch_first=False,
                          bidirectional=True)
        #self.gru = nn.GRU(input_size=local_size, hidden_size=128, batch_first=False,
        #                  bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.mu_layer = nn.Linear(256, local_size)
        self.lv_layer = nn.Linear(256, local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        # x shape : (window_size, batch, num_windows, voxels)
        window_size, batch_size, num_windows, voxels = x.size()
        x = x.view(window_size, batch_size * num_windows, voxels)
        features = self.lin(x)
        h_t, h_T = self.gru(features)
        h_T = torch.reshape(h_T.permute(1, 0, 2), (batch_size * num_windows, 256))
        h_T = self.dropout(h_T)
        zl_mean = self.mu_layer(h_T)
        zl_std = F.softplus(self.lv_layer(h_T))
        zl_mean = zl_mean.view(batch_size, num_windows, -1)
        zl_std = zl_std.view(batch_size, num_windows, -1)
        return D.Normal(zl_mean, zl_std)


class WindowDecoderRNN(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(WindowDecoderRNN, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(#nn.Linear(local_size + global_size, local_size + global_size), nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(local_size + global_size, 512), nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(512, 512), nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(512, 256))
        self.gru = nn.GRU(input_size=1, hidden_size=256)
        self.factor_layer = nn.Linear(256, local_size)
        self.lin = nn.utils.parametrizations.weight_norm(nn.Linear(local_size, output_size))
        self.dropout = nn.Dropout(0.2)

    def forward(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, num_windows, local_size = z_t.size()
        if z_g is not None:
            z = torch.cat((z_t, z_g.unsqueeze(1).repeat(1, num_windows, 1)), dim=-1)
        else:
            z = z_t
        z = z.view(batch_size * num_windows, z.size(-1))
        emb = self.embedding(z)
        emb = self.dropout(emb)
        # For each window ...
        in_x = torch.zeros((output_len, batch_size * num_windows, 1), device=z_t.device)       
        emb = torch.reshape(emb, (1, batch_size * num_windows, -1))
        h_t, _ = self.gru(in_x, emb)
        h_t = self.dropout(h_t)
        with torch.no_grad():
            self.factor_layer.weight.data = F.normalize(self.factor_layer.weight.data, dim=1)
        x = self.factor_layer(h_t)
        x = self.lin(x)#.permute(0, 2, 1)
        x = x.view(output_len, batch_size, num_windows, -1)
        return x

class mEncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(mEncoderGlobal, self).__init__()
        #self.lin = nn.utils.parametrizations.weight_norm(nn.Linear(input_size, global_size))
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.global_encoder_sp = nn.Sequential(
            nn.Linear(input_size, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.gru = nn.GRU(input_size=128, hidden_size=256)
        #hidden_sizes = [(256, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        #lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.ELU(inplace=True))
        #              for (in_s, out_s) in hidden_sizes]
        #self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(256, self.global_size)
        self.lv_layer = nn.Linear(256, self.global_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        batch_size, num_timesteps, voxels = x.size()
        x = x.view(batch_size * num_timesteps, voxels)
        x = self.global_encoder_sp(x)
        x = x.view(batch_size, num_timesteps, 128)
        x = x.permute(1, 0, 2)
        h_t, h_T = self.gru(x)
        h_T = h_t.mean(0)
        #h_T = torch.reshape(h_T.permute(1, 0, 2), (batch_size, 128))
        features = self.dropout(h_T)
        zg_mean = self.mu_layer(features)
        zg_std = F.softplus(self.lv_layer(features))
        return D.Normal(zg_mean, zg_std)
