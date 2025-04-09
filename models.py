import torch
from torch import nn

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class LSTMModel(nn.Module):

    def __init__(self, input_size, emb_size, lstm_hid_size, lstm_layers=1, segments=True):
        super().__init__()

        self.segments = segments

        self.emb_layer = nn.Linear(input_size, emb_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(emb_size, lstm_hid_size, num_layers=lstm_layers, batch_first=True)

        self.c_0 = nn.Parameter(torch.randn(1, lstm_layers, lstm_hid_size))
        self.h_0 = nn.Parameter(torch.randn(1, lstm_layers, lstm_hid_size))

        self.final_layer = nn.Linear(lstm_hid_size, 1)

    def forward(self, x):

        # The input has form (batch_size, seg_size, context_size)
        batch_size = x.shape[0]
        seg_size = x.shape[1]

        x = self.relu(self.emb_layer(x.reshape(-1, x.shape[-1]))).reshape(batch_size,seg_size,-1)
        
        output, _ = self.lstm(x, (self.h_0.repeat([1,batch_size,1]), self.c_0.repeat([1,batch_size,1])))
        
        return self.final_layer(self.relu(output))


class AttentionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len, n_layers = 2):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.emb_layer = nn.Linear(input_dim, hidden_dim)

        self.att_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                activation=nn.SiLU(),
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.final_layer = nn.Linear(seq_len * hidden_dim, 1)

    def forward(self, x):

        batch_size = x.shape[0]
        total_len = x.shape[1]

        indices = torch.from_numpy(sliding_window_view(np.arange(total_len), self.seq_len))

        # x.shape = (batch_size, seq_len, input_dim)
        x = self.emb_layer(x.view(-1, self.input_dim)).view(batch_size, total_len, -1)
        # x.shape = (batch_size, seq_len, hidden_dim)
        x = x[:,indices,:].view(-1,self.seq_len, self.hidden_dim)

        # Apply the Transformer layers
        for att_layer in self.att_layers:
            x = att_layer(x)

        # Pass x through the final layer
        x = self.final_layer(x.view((-1, self.seq_len * self.hidden_dim))).view((batch_size, -1, 1))

        result = torch.zeros((batch_size, total_len, 1))
        result[:, self.seq_len-1:] = x

        return result


class AttentionPMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len, pm_len, n_layers = 2):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pm_len = pm_len
        self.hidden_dim = hidden_dim

        self.persistent_memory = nn.Parameter(torch.randn((pm_len, self.hidden_dim)))

        self.emb_layer = nn.Linear(input_dim, hidden_dim)

        self.att_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                activation=nn.SiLU(),
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.final_layer = nn.Linear(seq_len * (pm_len + hidden_dim), 1)

        self.device = None

    def forward(self, x):

        batch_size = x.shape[0]
        total_len = x.shape[1]

        indices = torch.from_numpy(sliding_window_view(np.arange(total_len), self.seq_len))

        # x.shape = (batch_size, seq_len, input_dim)
        x = self.emb_layer(x.view(-1, self.input_dim)).view(batch_size, total_len, -1)
        # x.shape = (batch_size, seq_len, hidden_dim)
        x = x[:,indices,:].view(-1,self.seq_len, self.hidden_dim)

        # Concatenate persistent memory to the beginning
        pm_expanded = self.persistent_memory.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([pm_expanded, x], dim=1)

        # Apply the Transformer layers
        for att_layer in self.att_layers:
            x = att_layer(x)

        # Pass x through the final layer
        x = self.final_layer(x.view((-1, (self.seq_len + self.pm_len) * self.hidden_dim))).view((batch_size, -1, 1))

        result = torch.zeros((batch_size, total_len, 1)).to(self.device)
        result[:, self.seq_len-1:] = x

        return result