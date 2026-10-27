import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import *


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.05
MOMENTUM=0.9            # pytorch documentation often shows momentum of 0.9 as a default
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 10}



# TODO: need to find a clean way to pass the training standardization pixel_mean and pixel_std to the
# predict function's preparation. For now, we'll do it ugly since we always use same seed. 
# empirically observe these values and save them in code. 
PIXEL_MEANS = {"mnist": 0.130660, "cifar10": 0.473363, "imagenet32": 0.450095}
PIXEL_STDS = {"mnist": 0.308108, "cifar10": 0.251569, "imagenet32": 0.261117}

def prepare_train(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1].
    Standardize input based on mean and standard deviation as shown in classwork 5.1 notebook
    """
    X_tensor = torch.as_tensor(X).float() / 255.0
    pixel_mean = X_tensor.mean()
    pixel_std = X_tensor.std()
    X_tensor = (X_tensor - pixel_mean) / pixel_std
    return X_tensor, pixel_mean, pixel_std

def prepare_test(X, pixel_mean, pixel_std):
    """Convert uint8 images (0-255) to floats. Standardize using the pixel
    mean and pixel std found in training data. (We don't want to standardize based
    on test data.)"""
    X_tensor = torch.as_tensor(X).float() / 255.0
    X_tensor = (X_tensor - pixel_mean) / pixel_std
    return X_tensor

def create_model_for_task(task):
    #model = ResidualMLP(**TASKS[task])
    model = SwiGLUNetwork(**TASKS[task], task_name=task)
    return model



def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        _, predicted = logits.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    epoch_loss = total_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total
    return eval_loss, eval_acc
        


def _train_internal(task, X_train, y_train, seed=0, device="cuda", eval_data=None):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ
    model = create_model_for_task(task).to(device)

    # 2. Wrap the data so we can loop over it in shuffled mini batches
    prepared_training_data, pixel_mean, pixel_std = prepare_train(X_train)
    #print(f"task: {task}, pixel_mean: {pixel_mean:.6f}, pixel_std: {pixel_std:.6f}")
    dataset = TensorDataset(prepared_training_data, torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Choose the loss and optimizer
    loss_fn = nn.CrossEntropyLoss()         # expects raw logits
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
           
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for i in range(1, EPOCHS[task] + 1):
        train_loss, train_acc = train_epoch(model, loader, loss_fn, optimizer, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if eval_data is not None:
            X_test, y_test = eval_data
            X_test = prepare_test(X_test, pixel_mean, pixel_std) # !! using training pixel_mean and pixel_std
            test_dataset = TensorDataset(X_test, torch.as_tensor(y_test).long())
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(f"epoch {i}: train loss {train_loss:.4f}, train accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}")
        else:
            print(f"epoch {i}: train loss {train_loss:.4f}, train accuracy: {train_acc:.4f}")

    if eval_data is not None:
        return model, history
    else:
        return model


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
    """Original interface. Unchanged signature, returns only `model`."""
    return _train_internal(task, X_train, y_train, seed=seed, device=device, eval_data=None)

# ---------------------------------------------------------------------------
# 3. NEW WRAPPER FUNCTION (Passes test set & returns model + history)
# ---------------------------------------------------------------------------
def train_and_evaluate(task, X_train, y_train, X_test, y_test, seed=0, device="cuda"):
    """New interface for running training with continuous test logging."""
    return _train_internal(
        task, 
        X_train, 
        y_train, 
        seed=seed, 
        device=device, 
        eval_data=(X_test, y_test)
    )




def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    task_name = model.task_name
    pixel_mean = PIXEL_MEANS[task_name]
    pixel_std = PIXEL_STDS[task_name]
    X = prepare_test(X, pixel_mean, pixel_std)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)



