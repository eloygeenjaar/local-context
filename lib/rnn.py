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
        self.lin = nn.Linear(input_size, local_size)
        self.gru = nn.GRU(input_size=local_size, hidden_size=128, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.mu_layer = nn.Linear(256, self.local_size)
        self.lv_layer = nn.Linear(256, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        batch_size, input_size, window_size = x.size()
        x = x.permute(0, 2, 1)
        features = self.lin(x)
        h_t, h_T = self.gru(features)
        #h_T = torch.reshape(h_T.permute(1, 0, 2), (batch_size, 256))
        h_T = self.dropout(h_t.mean(1))
        zl_mean = self.mu_layer(h_T)
        zl_mean = zl_mean.view(batch_size, -1)
        zl_std = F.softplus(self.lv_layer(h_T))
        zl_std = zl_std.view(batch_size, -1)
        return D.Normal(zl_mean, zl_std)


class WindowDecoderRNN(nn.Module):
    def __init__(self, output_size, local_size, global_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(WindowDecoderRNN, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(local_size, 256)
        self.gru = nn.GRU(input_size=1, hidden_size=256, batch_first=True)
        self.factor_layer = nn.Linear(256, local_size)
        self.lin = nn.Linear(local_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z_t, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, local_size = z_t.size()
        emb = self.embedding(z_t)
        emb = self.dropout(emb)
        # For each window ...
        in_x = torch.zeros((batch_size, output_len, 1), device=z_t.device)       
        emb = torch.reshape(emb, (1, batch_size, -1))
        h_t, _ = self.gru(in_x, emb)
        h_t = self.dropout(h_t)
        with torch.no_grad():
            self.factor_layer.weight.data = F.normalize(self.factor_layer.weight.data, dim=1)
        x = self.factor_layer(h_t)
        x = self.lin(x).permute(0, 2, 1)
        return x
