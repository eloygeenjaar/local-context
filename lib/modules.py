import torch
from torch import nn
from torch import distributions as D


class LocalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ouput_size = output_size
        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=True)
        self.gru = nn.GRU(input_size=2 * hidden_size, hidden_size=hidden_size)
        self.local_mu = nn.Linear(hidden_size, output_size)
        self.local_lv = nn.Linear(hidden_size, output_size)
        self.prior_local = nn.GRU(input_size=1, hidden_size=hidden_size)
        self.prior_mu = nn.Linear(hidden_size, output_size)
        self.prior_lv = nn.Linear(hidden_size, output_size)
        self.bigru_h = nn.Parameter(torch.randn(2, 1, hidden_size))

    def forward(self, x, h_prior=None):
        num_timesteps, batch_size, _ = x.size()
        h_bigru = self.bigru_h.repeat(1, batch_size, 1)
        h_t, _ = self.bigru(x, h_bigru)
        # Append both forward and backward so that
        # they align on the same timesteps (flip the backward across time)
        local_features = torch.cat((
            h_t[..., :self.hidden_size],
            torch.flip(h_t[..., self.hidden_size:], dims=(0, ))), dim=-1)
        # Causal (time-wise, not as in causality) generation of the local
        # features
        local_features, _ = self.gru(local_features, h_prior)
        # Map to mean and standard deviation for normal distribution
        local_mu = self.local_mu(local_features)
        local_sd = (0.5 * self.local_lv(local_features)).exp()
        local_dist = D.Normal(local_mu, local_sd)
        # If we do not predict the prior, then just initialize
        # the initial state with zeros
        if h_prior is None:
            h_prior = torch.zeros(
                (1, batch_size, self.hidden_size), device=x.device)
        i_zeros = torch.zeros((num_timesteps, batch_size, 1), device=x.device)
        # Causal (time-wise, not as in causality) generation of the prior
        # the prior is generally learned for the local features
        ht_p, _ = self.prior_local(i_zeros, h_prior)
        prior_mu = self.prior_mu(ht_p)
        prior_sd = (0.5 * self.prior_lv(ht_p)).exp()
        prior_dist = D.Normal(prior_mu, prior_sd)
        return local_dist, prior_dist


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, local_size, context_size, hidden_size=256):
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.dropout = nn.Dropout(0.1)
        self.context_gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, 128), nn.ELU(), nn.Linear(128, 64))
        self.context_lin = nn.Linear(2 * hidden_size, 64)
        self.context_mu = nn.Linear(64, context_size)
        self.local_encoder = LocalEncoder(
            input_size + context_size, hidden_size, local_size)
        self.context_to_prior = nn.Linear(context_size, hidden_size)

    def get_context(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Find context representations
        _, hT_g = self.context_gru(x)
        hT_g = torch.reshape(
            hT_g.permute(1, 0, 2), (batch_size, 2 * self.hidden_size))
        hT_g = self.dropout(hT_g)
        hT_g = self.context_mlp(hT_g)
        context_mu = self.context_mu(hT_g)
        context_dist = D.Normal(context_mu, 0.1)
        context_z = context_dist.rsample()
        return context_dist, context_z

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Get context representations
        context_dist, context_z = self.get_context(x)
        # Use the original input and context representation to infer
        # each local representation
        context_z = context_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        context_input = torch.cat((x, context_z), dim=-1)
        h_prior = torch.zeros((1, batch_size, self.hidden_size),
                              device=x.device)
        local_dist, prior_dist = self.local_encoder(context_input, h_prior)
        local_z = local_dist.rsample()
        # The final latent representation is a concatenation of the
        # local and global representation
        z = torch.cat((local_z, context_z), dim=-1)
        return context_dist, local_dist, prior_dist, z


class LocalDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # Just a linear layer
        if self.num_layers == 1:
            self.layers.append(
                nn.Linear(input_size, output_size)
            )
        # Multiple layers with a linear layer at the end
        else:
            self.layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ELU()]
            )
            for _ in range(self. num_layers - 2):
                self.layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.ELU()]
                )
            self.layers.append(
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
