import argparse
import os
import os.path as osp

_BATCH_SIZE = 128
_EPOCHS = 350
_LR = 0.01


def get_train_parser(desc=""):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_path', type=str,
                        default=osp.join(osp.dirname(osp.abspath(__file__)), "dataset", "bin"),
                        help='Path to the dataset root directory')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu device to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=_BATCH_SIZE,
                        help=f'batch size for training and validation (default: {_BATCH_SIZE})')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=_EPOCHS,
                        help=f'number of epochs to train (default: {_EPOCHS})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'learning rate (default: {_LR})')
    parser.add_argument('--optimizer', type=str,
                        choices=('SGD', 'Adam'), default='Adam')

    parser.add_argument("--use-timestamp", action='store_true',
                        help='Whether to use timestamp in dump files')
    parser.add_argument("--times", type=int, default=1,
                        help='Number of times to run the experiment')
    parser.add_argument("--suffix", default='', type=str,
                        help='Suffix for the experiment name')
    return parser


def get_test_parser(desc=""):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--state', type=str, default=osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints", "uvnet_solidletters_chkpt.pt"),
                        help='PyTorch checkpoint file of trained network.')
    parser.add_argument('--no-cuda', action='store_true', help='Run on CPU')
    parser.add_argument('--subset', action='store_true',
                        help='Compute subset only (default: false)')
    parser.add_argument('--grams_path', type=str, required=True,
                        help='path to save Gram matrices to')
    return parser


def add_train_args(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=osp.join(osp.dirname(osp.abspath(__file__)), "data", "ABC", "bin"),
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Which gpu device to use (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_BATCH_SIZE,
        help=f"batch size for training and validation (default: {_BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_EPOCHS,
        help=f"number of epochs to train (default: {_EPOCHS})",
    )

    parser.add_argument(
        "--use-timestamp",
        action="store_true",
        help="Whether to use timestamp in dump files",
    )
    parser.add_argument(
        "--suffix", default="", type=str, help="Suffix for the experiment name"
    )
    return parser


def add_test_args(parser):
    parser.add_argument(
        "--state",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "uvnet_abc_chkpt.pt"),
        help="PyTorch checkpoint file of trainined network.",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Run on CPU")
    parser.add_argument("--seed", default=0, help="Seed")
    parser.add_argument("--grams_path",
                        default=osp.join(osp.dirname(osp.abspath(__file__)), "data", "ABC", "uvnet_grams"),
                        help="directory to save Gram matrices to (default: data/ABC/uvnet_grams)")
    return parser
