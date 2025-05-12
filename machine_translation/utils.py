import torch

def init_weights(model):
    """Initialize model weights."""
    for name, param in model.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)