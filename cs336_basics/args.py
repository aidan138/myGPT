from enum import Enum
from typing import Optional, IO

from pydantic import BaseModel, model_validator
import torch
import os


class ModelArgs(BaseModel):
    d_model: int
    vocab_size: int
    d_ff: Optional[float] = None
    rope_theta: float = 10000
    
    # Attention config
    num_layers: int
    num_heads: int = -1
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # Inference time parameters
    max_batch_size: int = 32
    max_seq_len: int = 2048 # Will be used at train as well but should be scaled down considerably

    @model_validator
    def validate(self) -> "ModelArgs":
        # Esssentially validating all attention heads
        assert self.num_kv_heads <= self.num_heads, f"Number of kv_heads: {self.num_kv_heads} must be less than n_heads: {self.num_heads}"
        assert self.num_heads % self.num_kv_heads == 0, f"n_heads {self.num_heads} must be divisible by {self.num_kv_heads}"
        assert self.dim % self.num_heads == 0, f"d_model: {self.d_model} must be divisible by n_heads: {self.num_heads}"
        return self
    

class TrainingArgs(BaseModel):
    # Train Loop
    iterations: int
    checkpoint_freq: int
    batch_size: int
    save_path: str | IO[bytes]
    train_path: str | IO[bytes]
    cv_path: str | IO[bytes]
    load_path: Optional[str | IO[bytes]]
    device: Optional[torch.device] = 'cpu'
    dtype: Optional[torch.dtype] = torch.float32

    # Logging
    log_cv_iterations: int
    log_train_iterations: int
    train_loss_alpha: float

    # Optimizer
    lr_max : float
    weight_decay: Optional [float] = None

    # Learning rate scheduler
    lr_min : Optional[float] = None
    warmup_iterations: Optional[float] = None
    cos_iterations: Optional[float] = None

    # Gradient Clipping
    max_l2_norm : Optional[float] = None
    


    def validate(self):
        assert self.iterations % self.checkpoint_freq, f"The checkpoint frequency ({self.checkpoint_freq}) must be divisible by the iterations ({self.iterations})"
        if self.lr_min or self.warmup_iterations or self.cos_iterations:
            assert None not in [self.lr_min, self.warmup_iterations, self.cos_iterations], f"If using annealing lr_min ({self.lr_min}), \
                  warmup_iterations ({self.warmup_iterations}), and cos_iterations ({self.cos_iterations}) must be set"

        return self