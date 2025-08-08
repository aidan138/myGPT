from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class ModelArgs(BaseModel):
    d_model: int
    vocab_size: int
    d_ff: Optional[float] = None
    rope_theta: float = 10000
    
    # Attention config
    n_layers: int
    n_heads: int = -1
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # Inference time parameters
    max_batch_size: int = 32
    max_seq_len: int = 2048 # Will be used at train as well but should be scaled down considerably

    @model_validator
    def validate(self) -> "ModelArgs":
        # Esssentially validating all attention heads
        assert self.n_kv_heads <= self.n_heads, f"Number of kv_heads: {self.n_kv_heads} must be less than n_heads: {self.n_heads}"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads {self.n_heads} must be divisible by {self.n_kv_heads}"
        assert self.dim % self.n_heads == 0, f"d_model: {self.d_model} must be divisible by n_heads: {self.n_heads}"
        return self