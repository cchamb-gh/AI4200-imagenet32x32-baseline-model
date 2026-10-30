import torch
from torch.utils.data import DataLoader


def run_branch_ablation(model: torch.nn.Module, val_loader: DataLoader, device: str) -> tuple[float, list[float]]:
    """
    Evaluates baseline accuracy, then sequentially zero-ablates each SwiGLU residual block's 
    branch to measure accuracy degradation.
    """
    model.eval()
    
    # Baseline Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            preds = model(x_val).argmax(dim=-1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)
            
    base_acc = correct / total
    acc_drops = []

    # Sequential block ablation via w_down output zeroing
    for idx, block in enumerate(model.blocks):
        def zero_branch_hook(module, inputs, output):
            return torch.zeros_like(output)

        # Updated hook registration target to w_down for SwiGLU blocks
        handle = block.w_down.register_forward_hook(zero_branch_hook)

        ablated_correct = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val).argmax(dim=-1)
                ablated_correct += (preds == y_val).sum().item()

        ablated_acc = ablated_correct / total
        acc_drops.append(base_acc - ablated_acc)
        
        handle.remove()

    return base_acc, acc_drops