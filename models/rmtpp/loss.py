import torch
import torch.nn as nn


def RMTPPLoss(historic_acc_info, time_duration):
    """Adopted from https://github.com/Hongrui24/RMTPP-pytorch/blob/master/model.py#L23
    """
    intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    # print('/3: ', torch.mean(torch.exp(intensity_w * time_duration)))
    loss = torch.mean(historic_acc_info \
                + intensity_w * time_duration
                + (torch.exp(historic_acc_info) -
                  torch.exp(historic_acc_info + intensity_w * time_duration)) / intensity_w)
    return -loss