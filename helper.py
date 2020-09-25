import logging
import torch
import os
import os.path as osp
import sys
import numpy as np
import random
from torch import optim
import platform


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not osp.exists(path):
        os.makedirs(path)


def save_checkpoint(
    filename: str,
    model,
    optimizer,
    scheduler=None,
    args=None,
    experiment_name: str = "",
):
    """
    Save model, optimizer and scheduler state to a file
    :param filename: Output state filename
    :param model: PyTorch model (torch.nn.Module)
    :param optimizer: PyTorch optimizer
    :param scheduler: PyTorch learning rate scheduler (default: None)
    :param args: Commandline args that was used for running experiment
    :param experiment_name: Name of the experiment
    :return: None
    """
    torch.save(
        {
            "model": model.state_dict(),
            "args": args,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "experiment_name": experiment_name,
        },
        "{}".format(filename),
    )


def load_checkpoint(filename: str):
    """
    Load model, optimizer and scheduler state from a file
    :param filename: Input state filename
    :return: Dictionary with 'model', 'optimizer', 'scheduler' and 'args' as keys and their states as values
    """
    return torch.load(filename)


def setup_logging(filename: str, level_str="info", filemode="w"):
    """
    Setup logging configuration
    :param filename: Log file
    :param level_str:
    :param filemode:
    :return:
    """
    if level_str == "error":
        level = logging.ERROR
    elif level_str == "warning":
        level = logging.WARNING
    elif level_str == "info":
        level = logging.INFO
    else:
        raise ValueError(
            'Unknown logging level {}. Expected one of ("error", "warning", "info")'
        )

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename, mode=filemode)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


def num_workers_platform():
    is_windows = any(platform.win32_ver())
    return 0 if is_windows else 4
