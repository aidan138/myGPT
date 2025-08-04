import torch
from torch import nn, Tensor
import math

class Linear(nn.Module):
    """
    Basic dense layer where every input feature has a corresponding weight within its output.
    """
    def __init__(self, in_features: int, output_features: int, bias: bool = False, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        dtype = torch.float32 if dtype is None else dtype
        print(f"The input features are: {in_features}, the output features are {output_features}")
        weight_stdv = math.sqrt(in_features + output_features)
        weights = nn.init.trunc_normal_(
            torch.zeros((output_features, in_features)),
            0, # Mean
            weight_stdv,
            -3 * weight_stdv, # Lower bound
            3 * weight_stdv # Uppper bound
        ).type(dtype)
        self.W = nn.Parameter(weights).to(device)
        self.b = nn.Parameter(torch.zeros(output_features)).to(device) if bias else None
        self.out = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W.T
        return out if self.b is None else out + self.b
    
    def get_num_parameters(self):
        return sum(param.numel() for param in self.parameters())

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype= None):
        super().__init__()
        dtype = torch.float32 if dtype is None else dtype
        embeddings = nn.init.trunc_normal_(
            torch.zeros((num_embeddings, embedding_dim)),
            0, # Mean
            1, # std
            -3, # Lower bound
            3 # Upper bound
        ).type(dtype)
        self.embeddings = nn.Parameter(embeddings).to(device)

    def forward(self, token_ids: Tensor):
        return self.embeddings[token_ids]
    
