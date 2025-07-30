import torch
from torch import nn, Tensor

class linear(nn.Module):
    """
    Basic dense layer where every input feature has a corresponding weight within its output.
    """
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        super().__init__()
        self.W = torch.randn(size=(fan_out, fan_in))
        self.b = torch.randn(fan_out) if bias else None
        self.out = None

    def forward(self, x: Tensor):
        self.out = self.W @ x + self.b
    
    def parameters(self):
        return [self.W, self.b] if self.b is not None else [self.W]
    
    def get_num_parameters(self):
        return sum(param.numel() for param in self.parameters())
    