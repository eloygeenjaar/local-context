import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F


class LocalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_val):
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
        self.dropout = nn.Dropout(dropout_val)

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
        local_features = self.dropout(local_features)
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
        ht_p = self.dropout(ht_p)
        prior_mu = self.prior_mu(ht_p)
        prior_sd = (0.5 * self.prior_lv(ht_p)).exp()
        prior_dist = D.Normal(prior_mu, prior_sd)
        return local_dist, prior_dist


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, local_size,
                 context_size, dropout_val, hidden_size=256,
                 independence=False):
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.independence = independence
        self.dropout = nn.Dropout(dropout_val)
        self.context_gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.context_mu = nn.Linear(2 * hidden_size, context_size)
        self.context_lv = nn.Linear(2 * hidden_size, context_size)
        self.local_encoder = LocalEncoder(
            input_size if independence else input_size + context_size,
            hidden_size, local_size, dropout_val=dropout_val)
        self.context_to_prior = nn.Linear(context_size, hidden_size)

    def get_context(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Find context representations
        _, hT_g = self.context_gru(x)
        hT_g = torch.reshape(
            hT_g.permute(1, 0, 2), (batch_size, 2 * self.hidden_size))
        hT_g = self.dropout(hT_g)
        context_mu = self.context_mu(hT_g)
        context_sd = F.softplus(self.context_lv(hT_g))
        context_dist = D.Normal(context_mu, context_sd)
        context_z = context_dist.rsample()
        return context_dist, context_z

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Get context representations
        context_dist, context_z = self.get_context(x)
        # Use the original input and context representation to infer
        # each local representation
        context_z = context_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        if self.independence:
            context_input = x
        else:
            context_input = torch.cat((x, context_z), dim=-1)
        h_prior = torch.zeros((1, batch_size, self.hidden_size),
                              device=x.device)
        local_dist, prior_dist = self.local_encoder(context_input, h_prior)
        local_z = local_dist.rsample()
        # The final latent representation is a concatenation of the
        # local and global representation
        z = torch.cat((local_z, context_z), dim=-1)
        return context_dist, local_dist, prior_dist, z

    def generate_counterfactual(self, x, cf_context):
        # The input to this function should essentiall look like this:
        # x: (num_timesteps, batch_size, data_size)
        # cf_context: (batch_size, context_size)
        # Each subject in the batch should be a subject we want to 
        # generate a counterfactual for, where
        # each entry in the cf_context batch should correspond
        # to the counterfactual context we should use for the 
        # corresponding subject in x.
        num_timesteps, batch_size, _ = x.size()
        context_z = cf_context.unsqueeze(0).repeat(num_timesteps, 1, 1)
        # Repeat along batch so we generate both non-cf and cf
        # The first batch_size subjects are non-cf
        # and the second batch_size entries are cfs
        if self.independence:
            context_input = x
        else:
            context_input = torch.cat((x, context_z), dim=-1)
        h_prior = torch.zeros((1, batch_size, self.hidden_size),
                              device=x.device)
        local_dist, prior_dist = self.local_encoder(context_input, h_prior)
        z = torch.cat((local_dist.mean, context_z), dim=-1)
        return local_dist, z


class LocalDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int, dropout_val: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.ELU
        # Just a linear layer
        if self.num_layers == 1:
            self.layers.append(
                nn.Linear(input_size, output_size)
            )
        # Multiple layers with a linear layer at the end
        else:
            self.layers.extend([
                nn.Linear(input_size, hidden_size),
                self.act(),
                nn.Dropout(dropout_val)]
            )
            for _ in range(self. num_layers - 2):
                self.layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    self.act(),
                    nn.Dropout(dropout_val)]
                )
            self.layers.append(
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvContextEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, window_size: int):
        super().__init__()
        # TODO: improve for future
        window_size = 32
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=4,
                      stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(hidden_size * 2 * (window_size // 4), hidden_size * 4),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, 2 * output_size)
        )

    def forward(self, x):
        x = F.pad(x, (1, 1))
        x = self.layers(x)
        mu, lv = torch.split(x, self.output_size, dim=-1)
        sd = torch.exp(0.5 * lv)
        return D.Normal(mu, sd)

class ConvContextDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, window_size: int):
        super().__init__()
        # TODO: improve for future
        window_size = 32
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, (window_size // 4) * hidden_size * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Unflatten(1, (hidden_size * 2, window_size // 4)),
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=4,
                      stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(hidden_size, output_size, kernel_size=4,
                      stride=2, padding=1),
        )

    def forward(self, x):
        # TODO improve
        return self.layers(x)[:, :, 1:-1]
