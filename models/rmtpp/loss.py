import torch
import torch.nn as nn


def RMTPPLoss(hidden_things, time_duration):
    """Adopted from https://github.com/Hongrui24/RMTPP-pytorch/blob/master/model.py#L23
    """
    intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    # print('/3: ', torch.mean(torch.exp(intensity_w * time_duration)))
    loss = torch.mean(hidden_things \
                + intensity_w * time_duration \
                + intensity_b \
                + (torch.exp(hidden_things + intensity_b) -
                  torch.exp(hidden_things + intensity_w * time_duration + intensity_b)) / intensity_w)
    return -loss