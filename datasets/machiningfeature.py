import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader
import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
import platform
import random
import helper
from sklearn.model_selection import train_test_split


def _collate(batch):
    graphs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    return bg, labels


class MachiningFeature(Dataset):
    def __init__(
        self, root_dir, split="train", size_percentage=1.0, apply_square_symmetry=0.0,
    ):
        """
        Load and the machining feature dataset from FeatureNet
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        :param size_percentage: Percentage of data to load per category (default: 1.0)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        """
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")

        files = list(path.rglob("*.bin"))
        print("Found {} {} data.".format(len(files), split))
        # Extract label from file name
        # e.g. 1 is the label for '1_2.SLDPRT.bin'
        labels = [int(f.name.split("_")[0]) for f in files]

        num_train = int(0.7 * len(files))
        num_val = int(0.15 * len(files))
        num_test = len(files) - num_train - num_val
        train_files, test_files, train_labels, test_labels = train_test_split(
            files, labels, test_size=0.15, random_state=42, stratify=labels
        )
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files,
            train_labels,
            test_size=0.2,
            random_state=84,
            stratify=train_labels,
        )
        if split == "train":
            self.graph_files = train_files
            self.labels = train_labels
        elif split == "val":
            self.graph_files = val_files
            self.labels = val_labels
        elif split == "test":
            self.graph_files = test_files
            self.labels = test_labels

        self.num_classes = 24
        self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
        print(
            "Done loading {} and data {} classes".format(
                len(self.graph_files), self.num_classes
            )
        )
        self.apply_square_symmetry = apply_square_symmetry

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.apply_square_symmetry > 0.0:
            prob_r = random.uniform(0.0, 1.0)
            if prob_r < self.apply_square_symmetry:
                graph.ndata["x"] = graph.ndata["x"].transpose(1, 2)
            prob_u = random.uniform(0.0, 1.0)
            if prob_u < self.apply_square_symmetry:
                graph.ndata["x"] = torch.flip(graph.ndata["x"], dims=[1])
            prob_v = random.uniform(0.0, 1.0)
            if prob_v < self.apply_square_symmetry:
                graph.ndata["x"] = torch.flip(graph.ndata["x"], dims=[2])

        return graph, torch.tensor([self.labels[idx]]).long()

    def feature_indices(self):
        """
        Returns a dictionary of mappings from the name of features to the channels containing them
        """
        return {
            "xyz": (0, 1, 2),
            "normals": (3, 4, 5),
            "mask": (6,),
            "E": (7,),
            "F": (8,),
            "G": (9,),
        }

    def get_dataloader(self, batch_size=128, shuffle=True):
        num_workers = helper.num_workers_platform()
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            num_workers=helper.num_workers_platform(),
        )


def _collate_pointclouds(batch):
    pcs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return pcs, labels


class MachiningFeaturePointclouds(Dataset):
    def __init__(self, root_dir, split="train", num_points=1024):
        """
        Load and the machining feature dataset from FeatureNet
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        :param num_points: Points per pointcloud
        """
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")

        files = list(path.rglob("*.npy"))
        print(f"Found {len(files)} samples.")
        # Extract label from file name
        # e.g. 1 is the label for '1_2.SLDPRT.bin'
        labels = [int(f.name.split("_")[0]) for f in files]

        num_train = int(0.7 * len(files))
        num_val = int(0.15 * len(files))
        num_test = len(files) - num_train - num_val
        train_files, test_files, train_labels, test_labels = train_test_split(
            files, labels, test_size=0.15, random_state=42, stratify=labels
        )
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files,
            train_labels,
            test_size=0.2,
            random_state=84,
            stratify=train_labels,
        )
        if split == "train":
            self.pc_files = train_files
            self.labels = train_labels
        elif split == "val":
            self.pc_files = val_files
            self.labels = val_labels
        elif split == "test":
            self.pc_files = test_files
            self.labels = test_labels

        self.num_classes = 24
        self.pointclouds = [np.load(fn) for fn in self.pc_files]
        print(
            "Done loading {} and data {} classes".format(
                len(self.pc_files), self.num_classes
            )
        )

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc = self.pointclouds[idx]
        return pc, torch.tensor([self.labels[idx]]).long()

    def get_dataloader(self, batch_size=128, shuffle=True):
        num_workers = helper.num_workers_platform()
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_pointclouds,
        )
