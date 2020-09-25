import argparse
import os.path as osp


_BATCH_SIZE = 128
_EPOCHS = 350


def add_train_args(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=osp.join(osp.dirname(osp.abspath(__file__)), "dataset"),
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
        default="",
        help="PyTorch checkpoint file of trainined network.",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Run on CPU")
    parser.add_argument("--seed", default=0, help="Seed")
    return parser
