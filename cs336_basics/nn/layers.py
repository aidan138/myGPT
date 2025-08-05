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
        # TODO Look into if they use running mean
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




class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0, "Input dim must be divisible by 2"
        self.sin, self.cos= self._compute_theta_mat(theta, d_k, max_seq_len, device)

    def _compute_theta_mat(self, theta: float, d_k: int, max_seq_len: int, device: torch.device):
        angle_vals = [
            i/(theta**(2*k/d_k)) for i in range(max_seq_len) for k in range(0, d_k//2)
        ]

        angle_vals = torch.Tensor(angle_vals).view((max_seq_len, d_k//2, -1)).to(device) # N, D/2, 1
        sin, cos = angle_vals.sin(), angle_vals.cos() # N, D/2, 1 for both
        return sin, cos

    def forward(self, x: Tensor, token_positions: torch.Tensor):
        *batch_dim, N, D = x.shape # Handle arbitrary batch dims
        x = x.view((*batch_dim, N, D//2, 2)).unsqueeze(-1) # B, N, D/2, 2, 1
        sin, cos = self.sin[token_positions], self.cos[token_positions] # N, D/2, 1
        R_mats = torch.stack( # N, D/2, 2 -> N, D/2, 2, 2
            (torch.concat([cos, -sin], dim=-1),
             torch.concat([sin, cos], dim=-1)), dim=-2
        ) 
        
        for _ in batch_dim:
            R_mats = R_mats.unsqueeze(0) # 1*B, N, D/2, 2,2
        return (R_mats @ x).squeeze(-1).view(*batch_dim, N, D)

