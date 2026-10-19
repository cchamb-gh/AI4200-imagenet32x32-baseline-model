import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TASKS, ResidualMLP
from .mnist_model import MNIST_Model


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.05
MOMENTUM=0.9            # pytorch documentation often shows momentum of 0.9 as a default
BATCH_SIZE = 256
EPOCHS = {"mnist": 20, "cifar10": 30, "imagenet32": 10}

def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------------------
# The two functions below are the REQUIRED INTERFACE for every submission.
# The grading notebook calls them like this:
#
#     model = train("cifar10", X_train, y_train, seed=0, device="cuda")
#     probs = predict(model, X_test)
#
# Change anything inside them, but keep their names and arguments.
# ---------------------------------------------------------------------------

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())

    # build task-specific models and optimizers
    if task == "mnist":
        model, optimizer, loader, loss_fn = train_mnist(dataset, device)
    elif task == "cifar10":
        model, optimizer, loader, loss_fn = train_cifar10(dataset, device)
    else:
        model, optimizer, loader, loss_fn = train_imagenet32(dataset, device)
            

    # The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

    return model


def train_mnist(dataset, device):
    """MNIST-specific training recipe"""
    width = 256
    depth = 13
    batch_size = BATCH_SIZE  # tune this later
    lr = LEARNING_RATE       # tune this later
    momentum = MOMENTUM      # tune this later

    model = MNIST_Model(**TASKS["mnist"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loader, loss_fn


def train_cifar10(dataset, device):
    model = ResidualMLP(**TASKS["cifar10"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    return model, optimizer, loader, loss_fn

def train_imagenet32(dataset, device):
    model = ResidualMLP(**TASKS["imagenet32"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    return model, optimizer, loader, loss_fn

def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)


def diag_input_variance(task, X_train, y_train, seed=0, device="cuda"):    
    torch.manual_seed(seed)                   # makes the run reproducible
    model = ResidualMLP(**TASKS[task]).to(device)
    X_fixed = X_train[:256]
    X_batch = prepare(X_fixed).to(device)

    # # 2. Wrap the data so we can loop over shuffled mini-batches.
    # dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # # 3. Choose the loss and the optimizer.
    # loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 4. The training loop.
    model.train()

    logits, stream = model(X_batch)           # forward pass
    variances = [layer_act.var().item() for layer_act in stream]
    means = [layer_act.mean().item() for layer_act in stream]
    pass

