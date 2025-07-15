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
    
class GRUModel(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_state = self.init_hidden()
        x, hidden = self.gru(x, hidden_state)
        out = self.fc(x[-1])
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(device='cuda' if torch.cuda.is_available() else 'cpu')