import os

import torch
import pandas as pd
from collections import Counter

from torch.utils.data import Dataset
import torch.nn.functional as F

class LoLDataset(Dataset):
    def __init__(self, input_dir, list_path):
        # Read input data
        games_list = pd.read_csv(list_path)
        self.input_dir = input_dir
        self.gids = games_list['gids']
        self.gfiles = games_list['gfiles']     

        self.seq_length = 6
        
    def __len__(self):
        return len(self.gfiles)

    def __getitem__(self, index):
        gid, gfile = self.gids[index], self.gfiles[index]
        df = pd.read_csv(os.path.join(self.input_dir, gfile))

        timestamp = torch.Tensor(df['timestamp'])
        timestamp -= torch.roll(timestamp, 1, 0)
        marker = torch.Tensor(df['event_type'])

        onehot = marker
        onehot = F.one_hot(onehot.long(), num_classes=2).float()
        return timestamp, onehot, marker