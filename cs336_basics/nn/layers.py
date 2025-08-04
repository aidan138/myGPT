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

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
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
    

class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, epsilon: float=1e-5, device = None, dtype= None):
        super().__init__()
        dtype = torch.float32 if dtype is None else dtype
        # TODO Look into if they ever use running mean
        self.gain = nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        assert D == self.gain.shape[0]
        input_dtype = x.dtype
        x = x.to(torch.float32) # Up cast to prevent overflow in lower dtypes
        square_sum_x = (x.square() + self.epsilon).sum(-1, keepdim=True) # B, N, 1
        rms_x = (square_sum_x / D).sqrt() # B, N, 1 
        # rms_x will be broadcasted across the feature dim and gain will be broadcasted across batch and sequence dims
        rms_norm_x = (x / rms_x) * self.gain # Element wise multiply and division

        return rms_norm_x.to(input_dtype)
    

def silu(x: Tensor):
    return x * torch.sigmoid(x)

class SwiGLU_Feedforward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        # Ensure the chosen dff is a multiple of 64 for CUDA
        d_ff = (d_ff // 64 + 1) * 64

        # The dimensionality of the weights
        dim_dict = {'W1': (d_model, d_ff),'W2': (d_ff, d_model), 'W3': (d_model, d_ff)}
        self.linears = nn.ModuleDict({
            layer_name: Linear(*dims, device=device, dtype=dtype) for layer_name, dims in dim_dict.items()
        })

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3
        xpreact = self.linears['W1'](x) # B, N, FF
        x_gate = self.linears['W3'](x)# B, N, FF
        # Apply elementwise multiplier with sigmoid activation
        # Note: sigmoid between 0 and 1
        x_silu = silu(xpreact)
        x_swiglu = x_silu * x_gate # Apply the linear transform gate
        return self.linears['W2'](x_swiglu)

