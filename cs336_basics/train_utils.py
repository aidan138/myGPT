import torch
import torch.nn as nn
from os import PathLike
from typing import BinaryIO
import numpy.typing as npt


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

def get_batch(dataset: npt.NDArray, batch_size, context_length, device):
    dataset = torch.from_numpy(dataset).to(device)
    document_length = dataset.shape[0] # Document length    
    choices = torch.randint(low=0, high=document_length-context_length, size=(batch_size,)) # Make 32 random choices from the document
    X, y = torch.stack([dataset[i: i+context_length] for i in choices], dim=0).to(device), torch.stack([dataset[i+1: i+context_length+1] for i in choices]).to(device)
    return (X, y)
