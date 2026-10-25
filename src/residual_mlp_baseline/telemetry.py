import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from .model import ResidualMLP, Block
#from .mnist_model import MNIST_FFN_Model, MNIST_FC_Layer


def compute_rms(tensor: torch.Tensor) -> float:
    """Computes Root-Mean-Square in float64 precision."""
    return torch.sqrt(torch.mean(tensor.double() ** 2)).item()


class ResidualMLPTelemetry:
    """
    Diagnostic telemetry engine custom-built for ResidualMLP.
    Extracts stream variance, branch-to-stream RMS ratios, layer gradients,
    activation saturation rates, and optimizer update ratios.
    """
    def __init__(self, model: ResidualMLP):
        self.model = model
        self.hooks = []
        
        # Telemetry containers
        self.stream_inputs: List[torch.Tensor] = []
        self.branch_outputs: List[torch.Tensor] = []
        self.fc1_pre_activations: List[torch.Tensor] = []

    def _attach_hooks(self):
        self._clear_containers()

        # Iterate over residual backbone blocks
        for block in self.model.blocks:
            # 1. Pre-hook on block to capture residual stream input x_l
            def make_stream_hook():
                def hook(module, inputs):
                    self.stream_inputs.append(inputs[0].detach().clone())
                return hook

            # 2. Hook on block.fc2 output to capture branch output f_l(x_l)
            def make_branch_hook():
                def hook(module, inputs, output):
                    self.branch_outputs.append(output.detach().clone())
                return hook

            # 3. Hook on block.fc1 output to capture pre-ReLU activations z
            def make_fc1_hook():
                def hook(module, inputs, output):
                    self.fc1_pre_activations.append(output.detach().clone())
                return hook

            self.hooks.append(block.register_forward_pre_hook(make_stream_hook()))
            self.hooks.append(block.fc2.register_forward_hook(make_branch_hook()))
            self.hooks.append(block.fc1.register_forward_hook(make_fc1_hook()))

    def _clear_containers(self):
        self.stream_inputs.clear()
        self.branch_outputs.clear()
        self.fc1_pre_activations.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def inspect_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> Dict[str, Any]:
        """Runs evaluation pass on a fixed diagnostic batch and computes key metrics."""
        self.model.eval()
        self._attach_hooks()

        # Forward pass
        logits = self.model(x)
        loss = loss_fn(logits, y)

        # Backward pass for gradients
        self.model.zero_grad()
        loss.backward()

        # Metric 1: Stream Variance & Metric 2: Branch-to-Stream RMS Ratio
        stream_variances = []
        branch_ratios = []
        for x_l, f_l in zip(self.stream_inputs, self.branch_outputs):
            # Variance across features, averaged over batch
            var = x_l.double().flatten(1).var(dim=1).mean().item()
            stream_variances.append(var)

            rms_stream = compute_rms(x_l)
            rms_branch = compute_rms(f_l)
            branch_ratios.append(rms_branch / (rms_stream + 1e-12))

        # Metric 3: Layer Weight Gradient RMS
        grad_rms_dict = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_rms_dict[name] = compute_rms(param.grad)

        # Metric 5: ReLU Saturation Rate (fraction of pre-activations <= 0)
        relu_saturation_rates = []
        for z in self.fc1_pre_activations:
            sat = (z <= 0.0).float().mean().item()
            relu_saturation_rates.append(sat)

        self.remove_hooks()
        self.model.train()

        return {
            "loss": loss.item(),
            "stream_variance": stream_variances,
            "branch_to_stream_ratio": branch_ratios,
            "weight_grad_rms": grad_rms_dict,
            "relu_saturation": relu_saturation_rates
        }


def compute_exact_update_ratios(
    model: ResidualMLP, 
    optimizer: torch.optim.Optimizer, 
    x_batch: torch.Tensor, 
    y_batch: torch.Tensor, 
    loss_fn: nn.Module
) -> Dict[str, float]:
    """
    Metric 4: Measures exact Update-to-Weight Ratio RMS(ΔW) / RMS(W) 
    over an SGD optimization step.
    """
    model.train()

    # Snapshot current weights
    w_before = {
        name: param.detach().clone() 
        for name, param in model.named_parameters() 
        if 'weight' in name
    }

    # Execute optimizer step
    optimizer.zero_grad()
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)
    loss.backward()
    optimizer.step()

    # Calculate step size relative to weight magnitude
    update_ratios = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                delta_w = param - w_before[name]
                rms_delta = compute_rms(delta_w)
                rms_w = compute_rms(w_before[name])
                update_ratios[name] = rms_delta / (rms_w + 1e-12)

    return update_ratios