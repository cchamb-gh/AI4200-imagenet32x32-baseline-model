import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#class MLP_Multiclass(nn.Module):
class MNIST_Block(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        #self.norm = nn.LazyBatchNorm1d()

    def forward(self, x):
        #branch = self.norm(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

class MNIST_Model(nn.Module):
    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=30):
        super().__init__()
        self.input_layer = nn.Linear(in_features, width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)
        
        self.blocks = nn.ModuleList([MNIST_Block(width, hidden) for _ in range(depth)])

    def forward(self, x):
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_fc2(torch.tanh(self.head_fc1(x)))
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)