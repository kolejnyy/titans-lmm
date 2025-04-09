import torch
import numpy as np
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

# Class for a time-series dataset
#   name:       the name of the dataset - ["sinwave", "weather"]
#   split:      the sata split - ["train", "valid", "test"]
#   segment:    whether to segment the sequence into k-element substrings
#   seg_size:   the number k for the splitting mentioned above

class TSDataset(Dataset):

    def __init__(self, name, split, segment = False, seg_size = 17):
        super().__init__()

        assert(name == 'sinwave' or name == 'weather')
        assert(split in ['train', 'valid', 'test'])
        self.name = name
        self.path = f'data/{name}/{split}.npy'

        # Raw numpy data and placeholder for torch tensor
        self.raw_data = np.load(self.path)
        self.data = None
        self.segment = segment
        self.process_data(segment, seg_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # If we chopped the sequence into segments, return separately the context and target
        if self.segment:
            if self.name == "sinwave":
                return self.data[index, :-1], self.data[index, -1]
            return self.data[index, :-1, :-1], self.data[index, -1, -1]
            
        # Otherwise, return the whole sequence's contexts and targets 
        if self.name == "sinwave":
            return self.data[index], self.data[index]
        return self.data[index, :-1], self.data[index, -1]

    def process_data(self, segment, seg_size):

        # If we do not want to cut the sequence into segments, just return the raw one
        _data = self.raw_data
        
        # Otherwise, segment the series, reshaping to size (N * (series_len - seg_size + 1), seg_size)
        if segment:
            if self.name == 'sinwave':
                _data = sliding_window_view(self.raw_data, seg_size, axis=1).reshape(-1, seg_size, 1)
            else:
                _data = np.permute_dims(sliding_window_view(self.raw_data, seg_size, axis=0), [0,2,1])
        
        # Convert from numpy to torch
        self.data = torch.from_numpy(_data)