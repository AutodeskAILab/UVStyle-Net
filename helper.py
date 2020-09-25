import logging
import torch
import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from mpl_toolkits.mplot3d import Axes3D
from torch import optim
from torch.utils.data import Dataset, DataLoader, RandomSampler


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


def get_optimizer(name: str, model, lr, **kwargs):
    """
    Create and initialize a PyTorch optimizer
    :param name: Name of the optimizer. One of ('SGD', 'Adam')
    :param model: PyTorch model (torch.nn.Module) whose parameters will be optimized
    :param lr: Learning rate
    :param kwargs: other keyword arguments to the optimizer
    :return: Optimizer
    """
    if name == "SGD":
        if lr is None:
            lr = 0.1
        return optim.SGD(model.parameters(), lr=lr, **kwargs)
    if name == "Adam":
        if lr is None:
            lr = 0.0001
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    raise ValueError("Unknown optimizer: " + name)


def save_checkpoint(filename: str, model, optimizer, scheduler=None, args=None):
    """
    Save model, optimizer and scheduler state to a file
    :param filename: Output state filename
    :param model: PyTorch model (torch.nn.Module)
    :param optimizer: PyTorch optimizer
    :param scheduler: PyTorch learning rate scheduler (default: None)
    :param args: Commandline args that was used for running experiment
    :return: None
    """
    torch.save(
        {'model': model.state_dict(), 'args': args, 
         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None},
        '{}'.format(filename))


def load_checkpoint(filename: str, map_to_cpu=False):
    """
    Load model, optimizer and scheduler state from a file
    :param filename: Input state filename
    :return: Dictionary with 'model', 'optimizer', 'scheduler' and 'args' as keys and their states as values
    """
    return torch.load(filename, map_location='cpu' if map_to_cpu else None)


def get_dataloader(dset, batch_size, train=True, collate_fn=None):
    """
    Returns the dataloader for the given dataset. Chooses num_workers carefully based on OS for best performance.
    :param dset PyTorch Dataset
    :param batch_size batch_size to use
    :param collate_fn Function to collate data into batches or None for default collate function in PyTorch
    :return PyTorch dataloader
    """
    import platform
    num_workers = 8
    if platform.system() == "Windows":
        num_workers = 0
    dataloader = DataLoader(dset,
                            batch_size=batch_size,
                            shuffle=train,
                            # sampler=RandomSampler(torch.arange(256)),
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True,)
    return dataloader


def setup_logging(filename: str, level_str="info", filemode="w"):
    """
    Setup logging configuration
    :param filename: Log file
    :param level_str:
    :param filemode:
    :return:
    """
    if level_str == 'error':
        level = logging.ERROR
    elif level_str == 'warning':
        level = logging.WARNING
    elif level_str == 'info':
        level = logging.INFO
    else:
        raise ValueError('Unknown logging level {}. Expected one of ("error", "warning", "info")')

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(filename, mode=filemode)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def network_plot_3D(G, angle, save=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()
    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    # if save is not False:
    #     plt.savefig("C:\scratch\\data\"+str(angle).zfill(3)+".png")
    #     plt.close('all')
    #     else:
    plt.show()

    return