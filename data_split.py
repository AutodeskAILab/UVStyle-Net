import pathlib
import dgl
from dgl.data.utils import load_graphs
from string import ascii_lowercase
from sklearn.model_selection import train_test_split


def solidmnist_split_by_num_nodes(root_dir, test_ratio=0.2):
    path = pathlib.Path(root_dir)
    bin_files = list(path.rglob("*.bin"))
    print(len(bin_files))
    num_nodes = []
    for fn in bin_files:
        g = load_graphs(str(fn))[0][0]
        num_nodes.append(g.number_of_nodes())
    easy = {k: [] for k in ascii_lowercase}
    medium = {k: [] for k in ascii_lowercase}
    hard = {k: [] for k in ascii_lowercase}
    # Create 3 bins for the graphs: easy, medium and hard based on the node count
    min_nodes = min(num_nodes)
    max_nodes = max(num_nodes)
    first_bin = int(min_nodes + 0.15 * (max_nodes - min_nodes))
    second_bin = int(min_nodes + 0.30 * (max_nodes - min_nodes))
    easy_range = (min_nodes, first_bin)
    medium_range = (first_bin, second_bin)
    hard_range = (second_bin, max_nodes)
    for i, filename in enumerate(bin_files):
        label = filename.name[0]
        if num_nodes[i] >= easy_range[0] and num_nodes[i] < easy_range[1]:
            easy[label].append(bin_files[i])
        elif num_nodes[i] >= medium_range[0] and num_nodes[i] < medium_range[1]:
            medium[label].append(bin_files[i])
        if num_nodes[i] >= hard_range[0] and num_nodes[i] < hard_range[1]:
            hard[label].append(bin_files[i])
    train_split = []
    test_split = []
    for label in ascii_lowercase:
        for jj, diff_map in enumerate((easy, medium, hard)):
            if len(diff_map[label]) > 0:
                # print(label, len(diff_map[label]))
                tr, te = train_test_split(
                    diff_map[label], test_size=test_ratio, random_state=42
                )
                train_split.extend(tr)
                test_split.extend(te)
    return train_split, test_split


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        "Split Solid/WireMNIST bin files into train/test splits"
    )
    parser.add_argument("root_dir", type=str, help="Root directory of dataset")
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of test set"
    )
    args, _ = parser.parse_known_args()
    train_files, test_files = solidmnist_split_by_num_nodes(
        root_dir=args.root_dir, test_ratio=args.test_ratio
    )
    with open(pathlib.Path(args.root_dir) / "train.txt", "w") as f:
        f.writelines([str(fn) for fn in train_files])
    with open(pathlib.Path(args.root_dir) / "test.txt", "w") as f:
        f.writelines([str(fn) for fn in test_files])
