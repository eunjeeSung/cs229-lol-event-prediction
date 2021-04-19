import torch
import torch.nn as nn
from torch.autograd import Variable 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMTPP(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                num_layers, batch_size, num_classes):
        super(RMTPP,self).__init__()
        self.embed = nn.Linear(input_size, embed_size)
        self.rnn = nn.RNN(input_size=hidden_size,
                        hidden_size=hidden_size,
                        batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers        
        self.batch_size = batch_size
        self.num_classes = num_classes
        
    def forward(self, timestamps, markers, hidden):
        embedded_markers = self.embed(markers)
        input_combined = torch.cat((timestamps, embedded_markers), axis=2)
        out, hidden = self.rnn(input_combined, hidden)
        logit = self.linear(out)
        return logit, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)