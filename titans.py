import torch
from torch import nn
from torch.func import functional_call
from torch.nn.functional import normalize

from neural_memory import NeuralMemory

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MACTitanLayer(nn.Module):

    def __init__(self, hidden_dim, seq_len, pm_len, n_layers_nmm = 2, alpha = 0.999, eta = 0.8, theta = 0.3):
        super().__init__()

        self.seq_len = seq_len
        self.pm_len = pm_len
        self.hidden_dim = hidden_dim
        self.inter_dim = (pm_len + 2 * hidden_dim)

        # Persistent memory weights
        self.persistent_memory = nn.Parameter(torch.randn((pm_len, self.hidden_dim)))

        # The attention-based processing core
        self.att_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            activation=nn.SiLU(),
            batch_first=True
        )

        # Mapping to queries for retrieving from NMM
        self.Q = nn.Linear(hidden_dim, hidden_dim)

        # The Neural Memory Module (NMM)
        self.nm_module = NeuralMemory(
            emb_dim = hidden_dim,
            n_layers = n_layers_nmm,
            hidden_dim = 2 * hidden_dim,
            alpha = alpha,
            eta = eta,
            theta = theta
        )

        self.final_layer = nn.Linear(self.inter_dim * hidden_dim, seq_len * hidden_dim)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.deivce = None

        self.outer_params = [self.persistent_memory] + list(self.Q.parameters()) + list(self.final_layer.parameters()) + list(self.att_layer.parameters())

    def forward(self, x):
        # x.shape = (batch_size, seq_len, hidden_dim)
        batch_size = x.shape[0]

        # Retrieve knowledge from the NMM
        queries = self.silu(normalize(self.Q(x.view(-1, self.hidden_dim))))
        nmm_vals = self.nm_module.retrieve(queries).view(batch_size, -1, self.hidden_dim)

        # Concatenate persistent and long-term memory to the beginning
        pm_expanded = self.persistent_memory.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([pm_expanded, nmm_vals, x], dim=1)

        # Pass through the attention layer
        x = self.silu(self.att_layer(x).view(-1, self.inter_dim * self.hidden_dim))
        x = self.final_layer(x).view(-1, self.hidden_dim)

        # Update the NMM
        _, new_params = self.nm_module.update(x)

        # Retrieve new information
        y = functional_call(self.nm_module, new_params, normalize(self.Q(x)))

        # Gate the output using the retrieved memory
        return (x * self.sigmoid(y)).view(batch_size, self.seq_len, self.hidden_dim)



class MACTitan(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        context_window,
        pm_len,
        n_layers = 2,
        n_layers_nmm = 2,
        alpha = 0.999,
        eta = 0.8,
        theta = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_window = context_window

        self.emb_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MACTitanLayer(
                hidden_dim,
                context_window,
                pm_len,
                n_layers_nmm=n_layers_nmm,
                alpha = alpha,
                eta = eta,
                theta = theta
            )
            for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(hidden_dim * context_window, output_dim)

        self.silu = nn.SiLU()

        self.outer_params = list(self.emb_layer.parameters()) + list(self.final_layer.parameters())
        for layer in self.layers:
            self.outer_params += layer.outer_params

    # Simple taking an input of shape (batch_size, context_len, input_dim)
    # and returns (batch_size, output_dim)
    def process(self, x):

        batch_size = x.shape[0]

        # Pass x through the embedding layer to get (batch_size, context_len, hidden_dim)
        x = self.emb_layer(x.reshape(-1, self.input_dim)).view(batch_size, self.context_window, self.hidden_dim)

        # Pass x thorugh all the MACTitanLayer's
        for layer in self.layers:
            x = x + self.silu(layer(x))

        return self.final_layer(x.view(batch_size, -1))


    def forward(self, x):
        # We are given a batch of long sequences (batch_size, N, input_dim)
        # The output should be of the size (batch_size, N, output_dim)
        res = torch.zeros((x.shape[0], x.shape[1], self.output_dim)).cuda()

        # Note that in MAC, to evaluate y_{t}, we must have processed chunks
        # x_{t-context_window:t}, x_{t-2*context_window:t-context_window} etc.

        # Let's then consider this problem as context_window subproblems, based
        # on the position index modulo context_window

        x = np.permute_dims(sliding_window_view(x.cpu(), self.context_window, axis=1), (0,1,3,2))
        residual = x.shape[1] % self.context_window

        stz = torch.from_numpy(x[:,:-residual].reshape(x.shape[0], -1, self.context_window, self.context_window, self.input_dim)).cuda()
        # stz: (batch_size, N//context_window, context_window, context_window, input_dim)

        for i in range(stz.shape[1]):
            slide = stz[:,i].reshape(-1, self.context_window, x.shape[-1])
            # slide: (batch_size * context_window, context_window, input_dim)
            out = self.process(slide).reshape(-1, self.context_window, self.output_dim)
            res[:, (i+1)*self.context_window -1:(i+2)*self.context_window -1] = out

        # residual_part: (batch_size, context_window - 1 + N % context_window, context_window, input_dim)
        residual_part = torch.from_numpy(x[:,-residual:].reshape(-1, self.context_window, self.input_dim)).cuda()
        res_out = self.process(residual_part).reshape(-1, residual, self.output_dim)
        res[:, -residual:] = res_out

        return res