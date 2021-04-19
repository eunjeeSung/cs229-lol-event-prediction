import os

import torch
import pandas as pd
from collections import Counter

from torch.utils.data import Dataset

class LoLDataset(Dataset):
    def __init__(self, input_dir, list_path):
        # Read input data
        games_list = pd.read_csv(list_path)
        self.input_dir = input_dir
        self.gids = games_list['gids']
        self.gfiles = games_list['gfiles']     

        self.window_size = 6
        self.gap_size = 1
        self.seq_length = self.window_size + self.gap_size - 1
        
    def __len__(self):
        return len(self.gfiles)
        # return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        gid, gfile = self.gids[index], self.gfiles[index]
        df = pd.read_csv(os.path.join(self.input_dir, gfile))
        df = df.drop(['gameId'], axis=1)
        df = df.select_dtypes(exclude=['object'])

        total_len = len(df)
        timeseries_num = total_len - self.seq_length + 1
        timeseries_x = []
        x_df = df.drop(['blueKill', 'redKill'], axis=1)
        for i in range(timeseries_num):
            timeseries_x.append(x_df[i:i+self.window_size].values)
        timeseries_x = torch.tensor(timeseries_x, dtype=torch.double) # timesereis_num x window_size x feature_nums
        timeseries_y = torch.tensor(df['blueKill'][self.seq_length-1:].values, dtype=torch.double) # timeseries_num x 1

        return timeseries_x, timeseries_y