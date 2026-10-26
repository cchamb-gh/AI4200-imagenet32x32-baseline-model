import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Each task differs only in its input size and its number of classes.
# The same model class is used for all three.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10,    # 28x28 grayscale
                   "width": 128, "hidden": 256, "depth": 30},
    
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10,    # 32x32 color
                   "width": 128, "hidden": 256, "depth": 30},
    
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000,  # 32x32 color
                   "width": 128, "hidden": 256, "depth": 30},
}


class Block(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width

        # --- From the LayerNormBlock template in the Week 5.1 Colab in class ----
        # --- 
        # He init. fc1 reds the normalized stream (not a ReLU output): variance 1 / fan_in
        nn.init.normal_(self.fc1.weight, mean=0.0, std=math.sqrt(1.0 / width))
        nn.init.zeros_(self.fc1.bias)

        # He init. fc2 reads a ReLU output: variance 2 / fan_in
        nn.init.normal_(self.fc2.weight, mean=0.0, std=math.sqrt(2.0 / hidden))
        nn.init.zeros_(self.fc2.bias)


    def forward(self, x):
        branch = self.norm(x)
        branch = self.fc1(branch)
        branch = F.relu(branch)
        branch = self.fc2(branch)
        
        return x + branch                          # skip connection: add the input back


class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=30):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)


        # # Xavier (Glorot) initialization for every linear layer, zero biases.
        # for layer in self.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        #         nn.init.zeros_(layer.bias)

    def forward(self, x, return_stream=False):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)
        stream = [x]
        for block in self.blocks:
            x = block(x)
            if return_stream:
                stream.append(x)
        # the head block
        # clay: by the time we get here in the original implementation, variance has grown to be over 30
        #   that's noteworthy but expected. 
        #   Therefore, we'll scale using the LayerNorm
        x = self.head_norm(x)
        x = self.head_fc1(x)
        #x = torch.tanh(x) # tanh gave 47.28% accuracy in baseline
        # x = torch.relu(x) # relu gave 50.18% accuracy
        x = torch.sigmoid(x)
        x = self.head_fc2(x)
        # x = self.head_fc2(torch.tanh(self.head_fc1(self.head_norm(x))))
        if return_stream:
            return x, stream
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)



class SwiGLUBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.w_gate = nn.Linear(width, hidden)
        self.w_up = nn.Linear(width, hidden)
        self.w_down = nn.Linear(hidden, width)

    def forward(self, x):
        # 1. GATE PATH: Project the input and apply the non-linear SiLU (Swish) function.
        # This creates a "gate" with values that decide how much information gets through.
        gate_signal = F.silu(self.w_gate(x))
        
        # 2. VALUE PATH: Project the input linearly (without an activation function).
        # This holds the actual feature values/data you want to process.
        value_data = self.w_up(x)
        
        # 3. GATING: Perform element-wise multiplication.
        # The gate signal selectively filters, scales, or blocks the feature values.
        gated_features = gate_signal * value_data
        
        # 4. DOWN PROJECTION: Compress the gated features back down to the model's original dimension.
        output = self.w_down(gated_features)

        return output
    
    

class SwiGLUNetwork(nn.Module):
    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=30):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([SwiGLUBlock(width, hidden) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x, return_stream=False):
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)
        stream = [x]
        for block in self.blocks:
            x = block(x)
            if return_stream:
                stream.append(x)
        x = self.head_norm(x)
        x = self.head_fc1(x)

        x = torch.sigmoid(x)
        x = self.head_fc2(x)
        if return_stream:
            return x, stream
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)

