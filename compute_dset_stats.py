from datasets.wiremnist import WireMNIST
from datasets.solidmnist import SolidMNIST
from datasets.machiningfeature import MachiningFeature
import argparse
import numpy as np


def compute_node_stats(dset):
    num_nodes = []
    for i, (data, _) in enumerate(dset.get_dataloader(1, False)):
        num_nodes.append(data.batch_num_nodes[0])
    print(f"Average nodes: {sum(num_nodes) / len(num_nodes)}")
    print(f"Min-max nodes: {min(num_nodes)}-{max(num_nodes)}")
    print(f"Median nodes: {np.median(num_nodes)}")


def compute_category_count(dset, normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np

    count = np.zeros(dset.num_classes)
    max_count = 0
    for (uvsolid, label) in dset:
        count[label] += 1
    if normalize:
        max_count = np.amax(count)
        count /= max_count
    print(count)
    plt.bar(np.arange(dset.num_classes), count)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset stats")
    parser.add_argument(
        "dataset",
        type=str,
        choices=("wiremnist", "solidmnist", "machiningfeature"),
        default=None,
    )
    parser.add_argument("dataset_path", type=str, default=None, help="Path to dataset")
    parser.add_argument(
        "--split", type=str, choices=("train", "val", "test"), help="Dataset split"
    )
    args, _ = parser.parse_known_args()

    dset = None
    if args.dataset == "wiremnist":
        dset = WireMNIST(args.dataset_path, split=args.split)
    elif args.dataset == "solidmnist":
        dset = WireMNIST(args.dataset_path, split=args.split)
    elif args.dataset == "machiningfeature":
        dset = MachiningFeature(args.dataset_path, split=args.split)
    compute_node_stats(dset)
    compute_category_count(dset)
