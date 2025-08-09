import torch
import torch.nn as nn
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.args import ModelArgs, TrainingArgs
from cs336_basics.nn.optim import AdamW
from cs336_basics.nn.utils import cross_entropy_loss, gradient_clipping, lr_cosine_scheduling
from cs336_basics.train_utils import load_checkpoint, save_checkpoint, get_batch
import numpy as np
import numpy.typing as npt
import wandb
import time
from dotenv import load_dotenv
import os

load_dotenv()
wandb_username = os.getenv('WANDBUSERNAME')
project_name = os.getenv('WANDBPROJECTNAME')


# Current args
# See args.py for more options
model_args = {
    # LM config
    'd_model': 128,
    'vocab_size': 10000,
    'd_ff': 256,
    'rope_theta': 10000,
    
    # Attention config
    'num_layers': 5,
    'n_heads': 32,

    #'n_kv_heads': Optional[int] = None,
    'head_dim': 8,

    # Inference time parameters
    'max_batch_size': 32,
    'max_seq_len': 256, # Will be used at train as well but should be scaled down considerably
}


train_args = {
    # Train Loop
    'iterations': 50,
    'checkpoint_freq': 100,
    'device': 'cuda' if torch.cuda.is_available else 'cpu',
    'dtype': torch.float32,
    'save_path': r'',
    'train_path': r'',
    'cv_path': r'',
    'load_path': None,

    # Optimizer
    'lr_max' : 0.001,
    'weight_decay': 0.01,

    # Learning rate scheduler
    'lr_min' : 1e-6,
    'warmup_iterations': None,
    'cos_iterations': None,

    # Gradient Clipping
    'max_l2_norm' : None, # for gradient clipping

    # Logging Parameters
    'log_cv_iterations': 10,
    'log_train_iterations': 10,
    'train_loss_alpha': 0.1
}


def get_cv_loss(model: nn.Module, val_set: npt.ArrayLike, iterations: int = 10):
    loss_total = 0
    for _ in range(iterations):
        X_cv, y_cv = get_batch(val_set)
        logits = model(X_cv)
        loss_total += cross_entropy_loss(logits, y_cv).item()
    return loss_total / iterations


def train(model: nn.Module, train_args: TrainingArgs, run: wandb.Run):
    
    optimizer = AdamW(
        params = model.parameters(),
        lr = train_args.lr,
        betas = train_args.betas,
        weight_decay = train_args.weight_decay
    )

    # Reload the provided checkpoint
    current_iter = 0 if train_args.load_path is None else load_checkpoint(train_args.load_path, model, optimizer)

    # mmep the file into memory for lazy batching
    train_set = np.memmap(train_args.train_path, np.uint16, 'r')
    val_set = np.memmap(train_args.cv_path, np.uint16, 'r')
    iterations, checkpoint_freq = train_args.iterations, train_args.checkpoint_freq
    log_cv_iterations, log_train_iterations = train_args.log_cv_iterations, train_args.log_loss_iterations
    best_cv = float('inf')
    running_loss = 0
    alpha = train_args.train_loss_alpha
    grad_clip = True if train_args.max_l2_norm else False

    for i in range(iterations):
        print(f"Starting iteration {i + current_iter}")

        # Forward pass
        X, y = get_batch(train_set)
        logits = model(X)
        loss = cross_entropy_loss(logits, y)
        running_loss = alpha * running_loss + (1 - alpha) * loss.item()

        # Perform an update step
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            gradient_clipping(model.parameters())

        optimizer.step()

        # Logging statistics
        if i + current_iter % log_cv_iterations == 0:
            model.eval()
            with torch.no_grad():
                cv_loss = get_cv_loss(model, val_set).item()
            print(f"CV loss at iteration {current_iter + i} is {cv_loss:.6f}")
            run.log({'cv_loss': cv_loss})

        if i + current_iter % log_train_iterations:
            print(f"Training loss at iteration {current_iter + i} is {running_loss:.6f}")
            run.log({'train_loss': running_loss})

        if checkpoint_freq % checkpoint_freq == 0:
            print(f"Checkpointing at iteration {current_iter + i}")
            save_checkpoint(model, optimizer, current_iter + i, train_args.save_path)
        
        break
            


def main(model_args: ModelArgs, train_args: TrainingArgs):

    transformer = TransformerLM(
        vocab_size = model_args.vocab_size,
        context_length = model_args.max_seq_len,
        num_layers = model_args.num_layers,
        num_heads = model_args.num_heads,
        d_model = model_args.d_model,
        d_ff = model_args.d_ff,
        head_dim = model_args.head_dim
    )

    run = wandb.init(
        entity = wandb_username,
        project = project_name,
        config = {'training': train_args.dump(), 'model': model_args.dump()}
    )

    print(f"Starting training")
    start_time = time.perf_counter()
    train(transformer, train_args, run)
    end_time = time.perf_counter()

    total_duration = end_time - start_time
    secs = total_duration
    hours = secs // 3600
    secs -= hours * 3600
    mins = secs // 60
    secs -= mins * 60
    print(f"Finished training\nTraining took {hours} hour(s) {mins} minutes and {secs:.2f} seconds")


if __name__ == '__main__':
    model_args = ModelArgs.validate(model_args)
    train_args = TrainingArgs.validate(train_args)
    assert model_args.max_batch_size >= train_args.batch_size
    main(model_args, train_args)