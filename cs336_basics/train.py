import torch
import torch.nn as nn
from os import PathLike
from typing import BinaryIO, IO
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.args import ModelArgs

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | PathLike | BinaryIO):

    state_dict = {
        'Model' : model.state_dict(),
        'Optimizer' : optimizer.state_dict(),
        'Iteration' : iteration
    }

    torch.save(state_dict, out)
    

def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['Model'])
    optimizer.load_state_dict(state_dict['Optimizer'])
    return state_dict['Iteration']

model_args = {
    'd_model': 128,
    'vocab_size': 10000,
    'd_ff': 256,
    'rope_theta': 10000,
    
    # Attention config
    'n_layers': 5,
    'n_heads': 64,
    #'n_kv_heads': Optional[int] = None,
    'head_dim': 8,

    # Inference time parameters
    'max_batch_size': 32,
    'max_seq_len': 256, # Will be used at train as well but should be scaled down considerably
}

        
