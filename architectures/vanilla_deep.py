import torch
import torch.nn as nn

# Define Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, reduction_factor=0.8):
        super(Generator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.tanh(self.fc_output(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, reduction_factor=0.8):
        super(Discriminator, self).__init__()
        layers = []
        current_size = hidden_size
        self.fc1 = nn.Linear(input_size, current_size)
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            next_size = int(current_size * reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            current_size = next_size
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_output = nn.Linear(current_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.sigmoid(self.fc_output(x))
        return x
