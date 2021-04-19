import torch
import torch.nn as nn
from torch.autograd import Variable 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size,
                num_layers, window_size):
    super(LSTM, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers)
    self.fc_1 =  nn.Linear(hidden_size, 128)
    self.fc = nn.Linear(128, num_classes)
    #self.softmax = nn.Softmax()

    self.relu = nn.ReLU()
    self.window_size = window_size

  def forward(self, x):
    h_0, c_0 = self._init_hidden_state(x)
    out, (h_n, c_n) = self.lstm(x.view(self.window_size, len(x), -1), (h_0, c_0))

    h_n = h_n.view(-1, self.hidden_size)
    out = self.relu(h_n)
    out = self.fc_1(out)
    out = self.relu(out)
    out = self.fc(out)
    #out = self.softmax(out)
   
    return out

  def _init_hidden_state(self, x):
    return (torch.zeros(self.num_layers, x.shape[0], self.hidden_size).double(),
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).double())