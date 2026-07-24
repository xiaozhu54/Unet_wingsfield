"""
Learning rate schedulers.
"""
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched


def build_scheduler(optimizer: optim.Optimizer, config):
    """
    Build a learning rate scheduler based on config.

    Default: StepLR (Reduce LR by gamma every step_size epochs)
    """
    return lr_sched.StepLR(
        optimizer,
        step_size=config.lr_scheduler_step,
        gamma=config.lr_scheduler_gamma,
    )
