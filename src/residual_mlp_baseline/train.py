import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.v2 as v2

from .model import *


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.

# for SGD, a larger learning rate is fine
#   LEARNING_RATE = 0.05
# But when we switch to AdamW, need to use a much smaller base learning rate.
SGD_LEARNING_RATE = 0.05
ADAMW_LEARNING_RATE = 0.001
LEARNING_RATE = ADAMW_LEARNING_RATE

MOMENTUM=0.9            # pytorch documentation often shows momentum of 0.9 as a default
BATCH_SIZE = 256
EPOCHS = {"mnist": 15, "cifar10": 30, "imagenet32": 10}


def get_transforms(task):
    if task == "mnist":
        train_transform = v2.Compose([
            v2.RandomResizedCrop(size=(28, 28), scale=(0.95, 1.05), antialias=True),
            v2.RandomRotation((-15, 15)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.25])
        ])
        eval_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.25])
        ])
    else:  # cifar10 or imagenet32
        # Experimenting with AutoAugment, which takes a policy value
        policy = v2.AutoAugmentPolicy.CIFAR10 if task == "cifar10" else v2.AutoAugmentPolicy.IMAGENET
        
        train_transform = v2.Compose([
            v2.AutoAugment(policy=policy),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.RandomResizedCrop(size=(32, 32), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            # normalizing with std of 0.25 spreads it across the range of [-2, 2]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            
        ])
        eval_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
             # normalizing with std of 0.25 spreads it across the range of [-2, 2]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])
    return train_transform, eval_transform


def format_tensor(X):
    X_tensor = torch.as_tensor(X)
    if X_tensor.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        X_tensor = X_tensor.unsqueeze(1)
    elif X_tensor.ndim == 4 and X_tensor.shape[-1] in (1, 3):  # (N, H, W, C) -> (N, C, H, W)
        X_tensor = X_tensor.permute(0, 3, 1, 2)
    return X_tensor

def create_model_for_task(task):
    model = SwiGLUNetwork(**TASKS[task], task_name=task)
    return model


def train_epoch(model, loader, loss_fn, optimizer, scheduler, transform, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = transform(X_batch.to(device))
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        _, predicted = logits.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f'current learning rate: {current_lr:.3e}')
    
    epoch_loss = total_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, transform, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = transform(inputs.to(device))
        targets = targets.to(device)
        #inputs, targets = inputs.to(device), targets.to(device)

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
    train_transform, eval_transform = get_transforms(task)

    X_train_formatted = format_tensor(X_train)
    dataset = TensorDataset(X_train_formatted, torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Choose the loss and optimizer
    loss_fn = nn.CrossEntropyLoss()         # expects raw logits
    
    # SGD optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=SGD_LEARNING_RATE, momentum=MOMENTUM)

    # AdamW optimizer is used commonly with SwiGLU networks. Let's give it a try.
    optimizer = torch.optim.AdamW(model.parameters(), lr=ADAMW_LEARNING_RATE, weight_decay=0.05)
    
    # Support AdamW optimizer with a scheduler to warm up the weighted averaging in the early epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS[task])
           
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for i in range(1, EPOCHS[task] + 1):
        train_loss, train_acc = train_epoch(model, loader, loss_fn, optimizer, scheduler, train_transform, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if eval_data is not None:
            X_test, y_test = eval_data
            #X_test = prepare_test(X_test, pixel_mean, pixel_std) # !! using training pixel_mean and pixel_std
            test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            test_loss, test_acc = evaluate(model, test_loader, loss_fn, eval_transform, device)
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
    _, eval_transform = get_transforms(task_name)

    X_formatted = format_tensor(X)

    chunks = []
    for i in range(0, len(X_formatted), 1024):
        batch = X_formatted[i:i + 1024]
        batch = eval_transform(batch.to(device))
        chunks.append(model.predict(batch).cpu())
    return torch.cat(chunks)



