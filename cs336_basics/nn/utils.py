import torch
from torch import Tensor
import math

def softmax(x: Tensor, dim: int = -1):
    norm_x = x-x.max(dim=dim, keepdim= True).values # Subtract the max for numeric stability
    softmax = norm_x.exp() / torch.sum(norm_x.exp(), dim=dim, keepdim=True)
    return softmax

def sdp_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None):
    batch, *rest, seq_len, feat_dim = Q.shape # 
    # Note works for a two different sequence lengths
    # Therefore you can decide to determine attention attending from two diff sequences
    # For a autoregressive SA it will always be same source though
    sim_matrix = Q @ K.transpose(-2,-1)/math.sqrt(feat_dim) # (B, *rest, N, D) @ (B, *rest, D, M) -> (B, *rest, N, M)
    if mask is not None:
        sim_matrix = torch.where(condition=mask, input=sim_matrix, other=float('-inf'))
    prob_matrix = softmax(sim_matrix, dim=-1)
    
    return  (prob_matrix @ V).view(batch, *rest, -1, feat_dim) # Returns B, ..., n, feat_dim
