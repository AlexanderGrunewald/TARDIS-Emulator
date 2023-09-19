import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # softplus
        x = self.fc2(x)
        return x