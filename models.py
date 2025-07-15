import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size+hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size+hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined_input = torch.cat((x, hidden_state), dim=1)
        hidden = torch.sigmoid(self.in2hidden(combined_input))
        out = self.in2output(combined_input)
        return hidden, out
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size+hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size+hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined_input = torch.cat((x, hidden_state), dim=1)
        hidden = torch.sigmoid(self.in2hidden(combined_input))
        out = self.in2output(combined_input)
        return hidden, out
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))