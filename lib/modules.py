import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, repeat, pack, unpack


class EncoderLocal(nn.Module):
    def __init__(self, input_size, local_size, window_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderLocal, self).__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=8 * hidden_size, batch_first=True,
                          bidirectional=True)
        # Create encoder_net
        hidden_sizes = [(16 * hidden_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
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
            h_T = torch.reshape(h_T.permute(1, 0, 2), (x.size(0), 16 * self.hidden_size))
            features = self.mlp(h_T)
            zl_mean.append(self.mu_layer(features))
            zl_std.append(F.softplus(self.lv_layer(features)))
            h_ts.append(h_t)
        zl_mean = torch.stack(zl_mean, dim=1)
        zl_std = torch.stack(zl_std, dim=1)
        h_t = torch.cat(h_ts, dim=1)
        return h_t, D.Normal(zl_mean, 0.1)

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
        return h_t, D.Normal(zg_mean, 0.1)

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

class ContEncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers, window_size, perc_mask):
        """Initializes the instance"""
        super(ContEncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=2 * hidden_size, batch_first=True)
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
        features = self.mlp(h_T.squeeze(0))
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
            nn.Linear(local_size + global_size, 16 * hidden_size)
            ])
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size * 16, batch_first=True)
        hidden_sizes = [(hidden_size * 16, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.LeakyReLU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        #self.factor_layer = nn.Linear(hidden_size * 8, local_size + global_size)
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
        #with torch.no_grad():
        #    self.factor_layer.weight.data = F.normalize(self.factor_layer.weight.data, dim=1)
        recon_seq = self.mlp(recon_seq)
        x_mean = self.mu_layer(recon_seq)
        return x_mean

class TransMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float =0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_val = dropout
        self.layers = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)

class Attention(nn.Module):
    def __init__(self, dim, head_size, heads = 8, dropout = 0.):
        super().__init__()
        inner_dim = head_size *  heads
        project_out = not (heads == 1 and head_size == dim)

        self.heads = heads
        self.scale = head_size ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.rotary_emb = RotaryEmbedding(dim=dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Last element is global token
        q = torch.cat((self.rotary_emb.rotate_queries_or_keys(q[:, :, :-1]),
                       q[:, :, -1].unsqueeze(2)), dim=2)
        k = torch.cat((self.rotary_emb.rotate_queries_or_keys(k[:, :, :-1]),
                       k[:, :, -1].unsqueeze(2)), dim=2)
        #q = self.rotary_emb.rotate_queries_or_keys(q)
        #k = self.rotary_emb.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Architecture based on: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers, num_timesteps):
        super().__init__()
        self.input_size = input_size
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.global_tokens = nn.Parameter((torch.randn(1, 1, hidden_size)))
        self.lin_embedding = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.Linear(self.input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.dropout = nn.Dropout(0.2)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                    Attention(hidden_size, 2 * hidden_size, heads = 8, dropout = 0.2),
                    TransMLP(hidden_size, 2 * hidden_size, dropout = 0.2)
                ]))
        
        self.mlp = TransMLP(hidden_size, 2 * hidden_size)
        self.mu_layer = nn.Linear(hidden_size, global_size)
        self.lv_layer = nn.Linear(hidden_size, global_size)
        

    def forward(self, x):
        batch_size, num_timesteps, input_size = x.size()

        x = self.lin_embedding(x)
        global_tokens = self.global_tokens.repeat(batch_size, 1, 1)
        x = torch.cat((x, global_tokens), dim=1)
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        global_tokens = x[:, -1]
        #global_tokens = x.mean(1)
        global_tokens = self.mlp(global_tokens)
        mu = self.mu_layer(global_tokens)
        sd = F.softplus(self.lv_layer(global_tokens))

        return None, D.Normal(mu, sd)

class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class convEncoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(convEncoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class upconv(nn.Module):
    def __init__(self, nin, nout):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class convDecoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(convDecoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output
