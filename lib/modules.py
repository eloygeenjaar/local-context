import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, repeat, pack, unpack


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, local_size, global_size):
        """Initializes the instance"""
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.global_size = global_size
        self.global_gru = nn.GRU(input_size=input_size, hidden_size=128,
                                 bidirectional=True)
        self.global_mu = nn.Linear(256, global_size)
        self.global_lv = nn.Linear(256, global_size) 
        self.bigru = nn.GRU(input_size=input_size + global_size, hidden_size=128,
                            bidirectional=True)
        self.local_gru = nn.GRU(input_size=256, hidden_size=128)
        self.local_mu = nn.Linear(128, local_size)
        self.local_lv = nn.Linear(128, local_size)
        self.prior_local = nn.GRU(input_size=1, hidden_size=128)
        self.prior_mu = nn.Linear(128, local_size)
        self.prior_lv = nn.Linear(128, local_size)

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        _, hT_g = self.global_gru(x)
        hT_g = torch.reshape(hT_g.permute(1, 0, 2), (batch_size, 256))
        global_mu = self.global_mu(hT_g)
        global_sd = (0.5 * self.global_lv(hT_g)).exp()
        global_dist = D.Normal(global_mu, global_sd)
        global_z = global_dist.rsample()
        global_z = global_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        h_t, _ = self.bigru(torch.cat((x, global_z), dim=-1))
        local_in = torch.cat((h_t[..., :128], torch.flip(h_t[..., 128:], dims=(0, ))), dim=-1)
        ht_l, _ = self.local_gru(h_t)
        local_mu = self.local_mu(ht_l)
        local_sd = (0.5 * self.local_lv(ht_l)).exp()
        local_dist = D.Normal(local_mu, local_sd)
        local_z = local_dist.rsample()
        i_zeros = torch.zeros((num_timesteps, 1, 1), device=x.device)
        h_zeros = torch.zeros((1, 1, 128), device=x.device)
        ht_p, _ = self.prior_local(i_zeros, h_zeros)
        prior_mu = self.prior_mu(ht_p)
        prior_sd = (0.5 * self.prior_lv(ht_p)).exp()
        prior_dist = D.Normal(prior_mu.repeat(1, batch_size, 1), prior_sd.repeat(1, batch_size, 1))
        z = torch.cat((local_z, global_z), dim=-1)
        return global_dist, local_dist, prior_dist, z

class TemporalEncoderC(nn.Module):
    def __init__(self, input_size, local_size, global_size):
        """Initializes the instance"""
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.global_size = global_size
        self.global_gru = nn.GRU(input_size=input_size, hidden_size=128,
                                 bidirectional=True)
        self.global_mu = nn.Linear(256, global_size)
        self.global_lv = nn.Linear(256, global_size) 
        self.enc = nn.Sequential(
            nn.Linear(input_size + global_size, 128), nn.LeakyReLU(True),
            nn.Linear(128, 128), nn.LeakyReLU(True))
        self.local_mu = nn.Linear(128, local_size)
        self.local_lv = nn.Linear(128, local_size)
        self.prior_local = nn.GRU(input_size=1, hidden_size=128)
        self.prior_mu = nn.Linear(128, local_size)
        self.prior_lv = nn.Linear(128, local_size)

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Pass input through bi-directional GRU
        _, hT_g = self.global_gru(x)
        # Final output is (2, batch_size, hidden_size // 2)
        hT_g = torch.reshape(hT_g.permute(1, 0, 2), (batch_size, 256))
        # Map final output to a mean and standard deviation
        global_mu = self.global_mu(hT_g)
        global_sd = (0.5 * self.global_lv(hT_g)).exp()
        # Instantiate a global distribution and sample
        global_dist = D.Normal(global_mu, global_sd)
        global_z = global_dist.rsample()
        # Repeat global samples across timesteps within a window
        global_z = global_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        # Concatenate the input and global samples together
        h_t = self.enc(torch.cat((x, global_z), dim=-1))
        # Map each timestep to a mean and standard deviation
        local_mu = self.local_mu(h_t)
        local_sd = (0.5 * self.local_lv(h_t)).exp()
        local_dist = D.Normal(local_mu, local_sd)
        local_z = local_dist.rsample()
        i_zeros = torch.zeros((num_timesteps, 1, 1), device=x.device)
        h_zeros = torch.zeros((1, 1, 128), device=x.device)
        ht_p, _ = self.prior_local(i_zeros, h_zeros)
        prior_mu = self.prior_mu(ht_p)
        prior_sd = (0.5 * self.prior_lv(ht_p)).exp()
        prior_dist = D.Normal(prior_mu.repeat(1, batch_size, 1), prior_sd.repeat(1, batch_size, 1))
        z = torch.cat((local_z, global_z), dim=-1)
        #z = torch.cat((torch.zeros_like(local_z), global_z), dim=-1)
        return global_dist, local_dist, prior_dist, z
