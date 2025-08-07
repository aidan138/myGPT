import torch
from torch.optim import Optimizer
from typing import Optional
import math
from collections.abc import Callable, Iterable



# **From Stanford for an example purposes**
class SGD(Optimizer):

    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f'Invalid lerning rate: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr'] # Get the learning rate
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1
        
        return loss
    
class AdamW(Optimizer):
    def __init__(self, param: Iterable[torch.nn.Parameter], lr: float = 0.001, betas: tuple = (0.9, 0.999), weight_decay: float = 0.01, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f'Invalid learning rate {lr}')
        
        defaults = {'lr': lr, 'betas': betas, 'decay': weight_decay, 'epsilon': eps}
        super().__init__(param, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, (b1, b2), decay, epsilon = group['lr'], group['betas'],  group['decay'], group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m, v, t = state.get('m', 0), state.get('v', 0), state.get('t', 1)
                m = b1 * m + (1 - b1) * grad
                # print(grad.shape)
                v = b2 * v + (1-b2) * grad**2
                # print(v)
                lr_t = lr * math.sqrt(1-b2**t)/(1-b1**t)
                p.data -= lr_t * m/(torch.sqrt(v) + epsilon)
                p.data -= lr*decay*p.data
                
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
                grad.free()


    
weights = torch.nn.Parameter(5 * torch.randn((10,10)))
opt = AdamW([weights], lr=1e1)

# for t in range(500):
#     opt.zero_grad() # Reset the gradients for learnable parameters
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients
#     opt.step() # Run optimizer step