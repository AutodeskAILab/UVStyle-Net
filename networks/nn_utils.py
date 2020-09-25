import torch
import torch.nn as nn
from networks.wconv import WeightedConv


def conv(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())


def conv1d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU())


def wconv(in_channels, out_channels, kernel_size, padding=0):
    return nn.Sequential(
        WeightedConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())


def fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU())
