import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_FC_Layer(nn.Module):
    """ One simple, fully connected hidden layer with a tanh activation.
    
    width out == width in
    """
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width)

        # using Xavier initialization, which is appropriate for a tanh activation
        nn.init.xavier_normal_(self.fc1.weight)
        # zero bias
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        return torch.tanh(self.fc1(x))

class MNIST_FFN_Model(nn.Module):
    """ A very simple fully connected feed forward MLP.

    Has some specified number of hidden layers that use tanh activation.

    the hidden param is ignored but kept in the init signature since our other tasks use it.
    And this class may use it in the future when we go back to residual MLP.

    """
    def __init__(self, in_features, num_classes, width=128, hidden=128, depth=5):

        # since this is a simple feed forward MLP that's fully connected, we will ignore hidden param and use only width
        super().__init__()
        self.input_layer = nn.Linear(in_features, width)
        self.output_layer = nn.Linear(width, num_classes)
        
        self.blocks = nn.ModuleList([MNIST_Block(width) for _ in range(depth)])

    def forward(self, x):
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_layer(torch.tanh(x))
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)