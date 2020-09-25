import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader
import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
import platform
import random
from datasets import font_util
import helper
from sklearn.model_selection import train_test_split


def _collate(batch):
    graphs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    return bg, labels


class SolidMNIST(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        size_percentage=None,
        in_memory=False,
        apply_square_symmetry=0.0,
        split_suffix="",
        center_and_scale=True,
    ):
        """
        Load and create the SolidMNIST dataset
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param split_suffix: Suffix for the split directory to use
        :param center_and_scale: Center and scale the UV grids
        """
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")
        if split == "train" or split == "val":
            subfolder = "train"
        else:
            subfolder = "test"

        path /= subfolder + split_suffix
        self.graph_files = list(
            x for x in path.glob("*.bin") if font_util.valid_font(x)
        )
        print("Found {} samples.".format(len(self.graph_files)))
        self.in_memory = in_memory
        self.center_and_scale = center_and_scale

        # The first character of filename must be the alphabet
        self.labels = [self.char_to_label(fn.stem[0]) for fn in self.graph_files]

        if split in ("train", "val"):
            train_files, val_files, train_labels, val_labels = train_test_split(
                self.graph_files,
                self.labels,
                test_size=0.2,
                random_state=42,
                stratify=self.labels,
            )
            if split == "train":
                self.graph_files = train_files
                self.labels = train_labels
            elif split == "val":
                self.graph_files = val_files
                self.labels = val_labels

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        self.num_classes = len(set(self.labels))
        if platform.system() == "Windows" or in_memory:
            print("Windows OS detected, storing dataset in memory")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
            if self.center_and_scale:
                for i in range(len(self.graphs)):
                    self.graphs[i].ndata["x"] = font_util.center_and_scale_uvsolid(
                        self.graphs[i].ndata["x"]
                    )
        print(
            "Done loading {} and data {} classes".format(
                len(self.graph_files), self.num_classes
            )
        )
        self.apply_square_symmetry = apply_square_symmetry

    def char_to_label(self, char):
        return ord(char.lower()) - 97

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        if platform.system() == "Windows" or self.in_memory:
            graph = self.graphs[idx]
        else:
            graph_file = str(self.graph_files[idx].absolute())
            graph = load_graphs(graph_file)[0][0]
            if self.center_and_scale:
                graph.ndata["x"] = font_util.center_and_scale_uvsolid(graph.ndata["x"])
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


def _collate_with_pointclouds(batch):
    graphs, pcs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return bg, pcs, labels


class SolidMNISTWithPointclouds(Dataset):
    def __init__(
        self,
        bin_root_dir,
        npy_root_dir,
        split="train",
        shape_type=None,
        size_percentage=None,
        in_memory=True,
        num_points=1024,
        apply_square_symmetry=0.0,
        split_suffix="",
        center_and_scale=True,
    ):
        """
        Load and create the SolidMNIST dataset
        :param bin_root_dir: Root path to the dataset of bin files
        :param npy_root_dir: Root path to the dataset of npy files
        :param split: Whether train or test set
        :param shape_type: Whether to load lower case or upper case characters. Must be 'lower' or 'upper'
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param split_suffix: Suffix for the split directory to use
        :param center_and_scale: Center and scale the UV grids
        """
        bin_path = pathlib.Path(bin_root_dir)
        npy_path = pathlib.Path(npy_root_dir)
        assert split in ("train", "val", "test")
        if split == "train" or split == "val":
            subfolder = "train"
        else:
            subfolder = "test"

        bin_path /= subfolder + split_suffix
        npy_path /= subfolder
        pointcloud_files = list(
            [x for x in npy_path.glob("*.npy") if font_util.valid_font(x)]
        )
        graph_files = list(
            [x for x in bin_path.glob("*.bin") if font_util.valid_font(x)]
        )
        print("Found {} {} data.".format(len(graph_files), subfolder))
        print("Found pointcloud {} {} data.".format(len(pointcloud_files), subfolder))

        self.in_memory = in_memory
        self.center_and_scale = center_and_scale

        # random.seed(1200)
        if split == "train":
            k = int(0.8 * len(graph_files))
            graph_files = random.sample(graph_files, k)
        elif split == "val":
            k = int(0.2 * len(graph_files))
            graph_files = random.sample(graph_files, k)

        if size_percentage != None:
            k = int(size_percentage * len(graph_files))
            graph_files = random.sample(graph_files, k)

        self.pc_hashmap = {}
        for file_name in pointcloud_files:
            # TODO: fix this weird extension and its handling
            query_name = file_name.name[:-8]  # remove .stl.npy extension
            self.pc_hashmap[query_name] = str(file_name)

        self.graph_files = []
        self.pc_files = []
        self.labels = []
        self.num_points = num_points

        for file_name in graph_files:
            query_name = file_name.name[:-4]  # remove .bin extension
            if shape_type is not None and shape_type not in query_name:
                continue
            if query_name not in self.pc_hashmap:
                # print("Error: ", query_name)
                continue
            self.graph_files.append(str(file_name))
            self.pc_files.append(self.pc_hashmap[query_name])
            self.labels.append(self.char_to_label(pathlib.Path(file_name).stem[0]))

        self.num_classes = len(set(self.labels))

        if platform.system() == "Windows" or self.in_memory:
            print("Storing dataset in memory")
            print("Loading graphs...")
            self.graphs = [load_graphs(fn)[0][0] for fn in self.graph_files]
            print("Loading pointclouds...")
            self.pointclouds = [torch.tensor(np.load(fn)) for fn in self.pc_files]
            if self.center_and_scale:
                for i in range(len(self.graphs)):
                    (
                        self.graphs[i].ndata["x"],
                        center,
                        scale,
                    ) = font_util.center_and_scale_uvsolid(
                        self.graphs[i].ndata["x"], return_center_scale=True
                    )
                    self.pointclouds[i] = (self.pointclouds[i] - center) * scale
        print(
            "Loaded {} face-adj graphs and {} pc files from {} classes".format(
                len(self.graph_files), len(self.pc_files), self.num_classes
            )
        )

        self.apply_square_symmetry = apply_square_symmetry

    def __len__(self):
        return len(self.graph_files)

    def char_to_label(self, char):
        return ord(char.lower()) - 97

    def __getitem__(self, idx):
        if platform.system() == "Windows" or self.in_memory:
            graph = self.graphs[idx]
            pc = self.pointclouds[idx][: self.num_points]
        else:
            graph_file = str(self.graph_files[idx])
            graph = load_graphs(self.graph_files[idx])[0][0]
            pointcloud_file = self.pc_files[idx]
            pc = torch.tensor(np.load(pointcloud_file)[: self.num_points])  # ['arr_0']
            if self.center_and_scale:
                graph.ndata["x"], center, scale = font_util.center_and_scale_uvsolid(
                    graph.ndata["x"], return_center_scale=True
                )
                pc = (pc - center) * scale
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
        label = torch.tensor([self.labels[idx]]).long()
        return graph, pc, label

    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_with_pointclouds,
            num_workers=helper.num_workers_platform(),
            drop_last=True,
        )


def _collate_pointclouds(batch):
    pcs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return pcs, labels


class SolidMNISTPointclouds(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        shape_type=None,
        size_percentage=None,
        in_memory=True,
        num_points=1024,
        split_suffix="",
        center_and_scale=True,
    ):
        """
        SolidMNIST pointclouds only
        :param root_dir: Root path to the dataset of npy files
        :param split: Whether train or test set
        :param shape_type: Whether to load lower case or upper case characters. Must be 'lower' or 'upper'
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param split_suffix: Suffix for the split directory to use
        :param center_and_scale: Center and scale the pointclouds
        """
        npy_path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")
        if split == "train" or split == "val":
            subfolder = "train"
        else:
            subfolder = "test"

        npy_path /= subfolder
        pointcloud_files = list(
            [x for x in npy_path.glob("*.npy") if font_util.valid_font(x)]
        )
        if size_percentage is not None:
            random.seed(1200)
            k = int(size_percentage * len(pointcloud_files))
            pointcloud_files = random.sample(pointcloud_files, k)
        labels = [
            self.char_to_label(pathlib.Path(fn).stem[0]) for fn in pointcloud_files
        ]
        print("Found {} pointclouds.".format(len(pointcloud_files)))

        self.in_memory = in_memory
        self.center_and_scale = center_and_scale

        if split in ("train", "val"):
            train_files, val_files, train_labels, val_labels = train_test_split(
                pointcloud_files,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
            if split == "train":
                self.pc_files = train_files
                self.labels = train_labels
            elif split == "val":
                self.pc_files = val_files
                self.labels = val_labels
        else:
            self.pc_files = pointcloud_files
            self.labels = labels

        self.num_points = num_points
        self.num_classes = len(set(self.labels))

        if platform.system() == "Windows" or self.in_memory:
            print("Storing dataset in memory")
            print("Loading pointclouds...")
            self.pointclouds = [torch.tensor(np.load(fn)) for fn in self.pc_files]
            if self.center_and_scale:
                self.pointclouds = [
                    font_util.center_and_scale_pointcloud(pc) for pc in self.pointclouds
                ]
        print(
            "Loaded {} pc files from {} classes".format(
                len(self.pc_files), self.num_classes
            )
        )

    def __len__(self):
        return len(self.pc_files)

    def char_to_label(self, char):
        return ord(char.lower()) - 97

    def __getitem__(self, idx):
        if platform.system() == "Windows" or self.in_memory:
            pc = self.pointclouds[idx][: self.num_points]
        else:
            pointcloud_file = self.pc_files[idx]
            pc = torch.tensor(np.load(pointcloud_file)[: self.num_points])
            if self.center_and_scale:
                pc = font_util.center_and_scale_pointcloud(pc)
        label = torch.tensor([self.labels[idx]]).long()
        return pc, label

    def get_dataloader(self, batch_size=128, shuffle=True):
        num_workers = helper.num_workers_platform()
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_pointclouds,
            num_workers=num_workers,
        )
