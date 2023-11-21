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
        # Create encoder_net
        #hidden_sizes = [(input_size, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        #lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.ELU(inplace=True), nn.Dropout(0.05))
        #              for (in_s, out_s) in hidden_sizes]
        #self.mlp = nn.Sequential(*lin_layers)
        #self.ml
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=128, batch_first=True,
                          bidirectional=True)
        #elf.mlp = nn.Linear(1024, hidden_size)
        self.mu_layer = nn.Linear(256, self.local_size)
        self.lv_layer = nn.Linear(256, self.local_size)

    def forward(self, x, mask=None, window_step=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        zl_mean, zl_std, h_ts = [], [], []
        if window_step is None:
            window_step = self.window_size
        # x_w is size: (batch_size, num_windows, input_size, window_size)
        x_w = x.unfold(1, self.window_size, window_step)
        # Permute to: (batch_size, num_windows, window_size, input_size)
        x_w = x_w.permute(0, 1, 3, 2)
        batch_size, num_windows, window_size, input_size = x_w.size()
        x_w = x_w * mask
        x_w = torch.reshape(x_w, (batch_size * num_windows, window_size, input_size))
        x_w -= x_w.mean(0)
        x_w /= x_w.std(0)
        features = self.mlp(x_w)
        h_t, h_T = self.gru(features)
        h_T = torch.reshape(h_T.permute(1, 0, 2), (x_w.size(0), 256))
        zl_mean = self.mu_layer(h_T)
        zl_mean = zl_mean.view(batch_size, num_windows, self.local_size)
        h_t = h_t.view(batch_size, num_windows, window_size, 256)
        return h_t, D.Normal(zl_mean, 0.01)

class EncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(EncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=256, batch_first=True)
        hidden_sizes = [(256, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.ELU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        h_t, h_T = self.gru(x)
        features = self.mlp(h_T.squeeze(0))
        zg_mean = self.mu_layer(features)
        return h_t, D.Normal(zg_mean, 0.1)

class mEncoderGlobal(nn.Module):
    def __init__(self, input_size, global_size, hidden_size, num_layers):
        """Initializes the instance"""
        super(mEncoderGlobal, self).__init__()
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=128, bidirectional=True)
        hidden_sizes = [(256, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.ELU(inplace=True))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        self.mu_layer = nn.Linear(hidden_size, self.global_size)
        self.lv_layer = nn.Linear(hidden_size, self.global_size)

    def forward(self, x):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        h_t, h_T = self.gru(x)
        features = self.mlp(h_t).mean(1)
        zg_mean = self.mu_layer(features)
        return D.Normal(zg_mean, 0.1)

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
            nn.Linear(local_size + global_size, 256, bias=False),
            ])
        self.gru = nn.GRU(input_size=1, hidden_size=256, batch_first=True)
        hidden_sizes = [(256, hidden_size)] + [(hidden_size, hidden_size)] * (num_layers - 1)
        lin_layers = [nn.Sequential(nn.Linear(in_s, out_s), nn.ELU(inplace=True), nn.Dropout(0.05))
                      for (in_s, out_s) in hidden_sizes]
        self.mlp = nn.Sequential(*lin_layers)
        #self.factor_layer = nn.Linear(1024, local_size + global_size, bias=False)
        self.mu_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        batch_size, num_windows, local_size = z_t.size()
        if z_g is not None:
            z = torch.cat((z_t, z_g.unsqueeze(1).repeat(1, num_windows, 1)), dim=-1)
        else:
            z = z_t
        emb = self.embedding(z)
        # For each window ...
        in_x = torch.zeros((batch_size * num_windows, output_len, 1), device=z_t.device)       
        emb = torch.reshape(emb, (1, batch_size * num_windows, -1))
        h_t, _ = self.gru(in_x, emb)
        h_t = self.mlp(h_t)
        x_mean = self.mu_layer(h_t)
        x_mean = x_mean.view(batch_size, num_windows * output_len, -1)
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
        #sd = 0.01
        return D.Normal(mu, sd)
