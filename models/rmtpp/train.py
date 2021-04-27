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
# from rmtpp import RMTPP
from rmtpp_lstm import RMTPP
from loss import RMTPPLoss


def get_loss(model, timestamps, onehots, markers):
    hidden = model.init_hidden()    
    timestamps, onehots, markers = timestamps.unsqueeze(2).to(device), \
                                    onehots.to(device), \
                                    markers.to(device)

    try:
        scores, intensity, hidden = model(timestamps[:, :-1], onehots[:, :-1], hidden)
    except Exception as err:
        #print(f'Epoch {epoch}, gid {gids}: err - {err}')
        return None, None, None
    preds_timestamp = scores[:, :, 0].unsqueeze(2)
    preds_marker = scores[:, :, 1:].permute(0, 2, 1)

    timestamp_loss = RMTPPLoss(preds_timestamp, timestamps[:, 1:])
    marker_loss = marker_criterion(preds_marker, markers[:, 1:].long())      

    loss = timestamp_loss + marker_loss

    return loss, marker_loss, timestamp_loss

def validate(model, test_dataloader):
    val_losses, val_marker_losses, val_timestamp_losses = [], [], []
    for batch_idx, (gids, timestamps, onehots, markers) in enumerate(tqdm(test_dataloader)):
        loss, marker_loss, timestamp_loss = get_loss(model, timestamps, onehots, markers)
        if loss is None:
            continue

        val_losses.append(loss)
        val_marker_losses.append(marker_loss)
        val_timestamp_losses.append(timestamp_loss)
    
    val_loss = sum(val_losses) / len(val_losses)
    val_marker_loss = sum(val_marker_losses) / len(val_marker_losses)
    val_timestamp_loss = sum(val_timestamp_losses) / len(val_timestamp_losses)
    return val_loss, val_marker_loss, val_timestamp_loss


if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 100000
    learning_rate = 0.0003

    num_classes = 3 # binary marker class + timestamp
    input_size = 2 # one-hot size
    embed_size = 5
    hidden_size = embed_size + 1 # output from the cell
    batch_size = 1
    num_layers = 1    

    # Configuration
    cfg = configparser.ConfigParser()
    cfg.read('../../config.cfg') 
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = LoLDataset(cfg['rmtpp']['input_dir'] , cfg['rmtpp']['games_list'])
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])    
    
    train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            drop_last=True)
    val_dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            drop_last=True)

    # Model
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
        for batch_idx, (gids, timestamps, onehots, markers) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            # Get training loss
            loss, marker_loss, timestamp_loss = get_loss(model, timestamps, onehots, markers)
            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                #print("Epoch: %d, marker loss: %1.5f, timestamp loss: %1.5f" \
                #        % (epoch, marker_loss.item(), timestamp_loss.item()))
                
                # Get validation loss
                val_loss, val_marker_loss, val_timestamp_loss = validate(model, val_dataloader)                

                writer.add_scalar('Loss/train', loss, i)
                writer.add_scalar('Marker Loss/train', marker_loss, i)
                writer.add_scalar('Timestamp Loss/train', timestamp_loss, i)

                writer.add_scalar('Loss/val', val_loss, i)
                writer.add_scalar('Marker Loss/val', val_marker_loss, i)
                writer.add_scalar('Timestamp Loss/val', val_timestamp_loss, i)                
                writer.flush()
            i += 1
    
    writer.close()