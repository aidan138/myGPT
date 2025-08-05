import torch
from torch import Tensor


def softmax(x: Tensor, dim: int):
    norm_x = x-x.max(dim=dim, keepdim= True).values # Subtract the max for numeric stability
    softmax = norm_x.exp() / torch.sum(norm_x.exp(), dim=dim, keepdim=True)
    return softmax