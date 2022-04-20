import torch.nn as nn

class MLP4(nn.Module):
    def __init__(self):
        super(MLP4, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x