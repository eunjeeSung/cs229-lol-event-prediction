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


from data import LoLDataset
from rmtpp import RMTPP
from loss import RMTPPLoss

if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read('../../config.cfg') 
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LoLDataset(cfg['file']['sample_dir'] , cfg['file']['sample_list'])
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            drop_last=True)

    # Hyperparameters
    num_epochs = 100000
    learning_rate = 0.1

    num_classes = 3 # binary marker class + timestamp
    input_size = 2 # one-hot size
    embed_size = 5
    hidden_size = embed_size + 1 # output from the cell
    batch_size = 2
    num_layers = 1

    model = RMTPP(input_size,
                embed_size,
                hidden_size,
                num_layers,
                batch_size,
                num_classes)
    model.to(device)

    # timestamp_criterion = torch.nn.MSELoss()
    marker_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    i = 0
    # # # Training Loop
    for epoch in range(num_epochs):
        for _, (timestamps, onehots, markers) in enumerate(dataloader):
            timestamps, onehots, markers = timestamps.unsqueeze(2).to(device), \
                                            onehots.to(device), \
                                            markers.to(device)

            optimizer.zero_grad()
            hidden = model.init_hidden()

            scores, intensity, hidden = model(timestamps[:, :-1], onehots[:, :-1], hidden)
            preds_timestamp = scores[:, :, 0].unsqueeze(2)
            preds_marker = scores[:, :, 1:].permute(0, 2, 1)

            timestamp_loss = RMTPPLoss(preds_timestamp, timestamps[:, 1:])
            marker_loss = marker_criterion(preds_marker, markers[:, 1:].long())      

            loss = timestamp_loss + marker_loss

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("Epoch: %d, marker loss: %1.5f, timestamp loss: %1.5f" \
                        % (epoch, marker_loss.item(), timestamp_loss.item()))
                writer.add_scalar('Loss/train', loss, i)
                writer.flush()
            i += 1
    
    writer.close()