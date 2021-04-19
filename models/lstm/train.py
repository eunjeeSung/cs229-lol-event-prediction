import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable 

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


from lstm import LSTM
from data import LoLDataset

if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read('../../config.cfg') 
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LoLDataset(cfg['file']['input_dir'] , cfg['file']['games_list'])
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            drop_last=True)

    # # Hyperparameters
    num_epochs = 300
    learning_rate = 0.00001
    input_size = 29
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    window_size = 6

    model = LSTM(num_classes=num_classes,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                window_size=window_size)
    model.to(device)
    model = model.double()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    i = 0
    # # Training Loop
    for epoch in range(num_epochs):
        for _, (ts_x, ts_y) in enumerate(tqdm(dataloader)):
            ts_x, ts_y = ts_x[0].to(device), ts_y[0].to(device)

            optimizer.zero_grad()

            outputs = model.forward(ts_x)
            loss = criterion(outputs[:, 0], ts_y)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                #print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
                writer.add_scalar('Loss/train', loss, i)
                writer.flush()
            i += 1
    
    writer.close()