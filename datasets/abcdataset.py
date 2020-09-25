import pathlib
import random

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import helper
from datasets import font_util


def _collate(batch):
    graphs = batch
    bg = dgl.batch(graphs)
    return (
        bg,
        None,
    )  # Return None for labels so that we can reuse the same train scripts


class ABCDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split="train",
            size_percentage=None,
            in_memory=True,
            apply_square_symmetry=0.0,
    ):
        """
        ABC dataset with both solids only
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val, test or all (entire) set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param split_suffix: Suffix for the split directory to use
        """
        assert split in ("train", "val", "test", "all")
        path = pathlib.Path(root_dir)
        self.graph_files = list(path.glob("*.bin"))
        self.labels = []

        self.in_memory = in_memory

        random.seed(1200)
        if split == "train":
            k = int(0.6 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "val":
            k = int(0.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "test":
            k = int(0.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "all":
            graph_files = graph_files

        print("Found {} {} data.".format(len(self.graph_files), split))

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        if in_memory:
            print("Windows OS detected, storing dataset in memory")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
            if self.center_and_scale:
                for i in range(len(self.graphs)):
                    self.graphs[i].ndata["x"] = font_util.center_and_scale(
                        self.graphs[i].ndata["x"]
                    )
        print(f"Done loading {len(self.graph_files)} data")
        self.apply_square_symmetry = apply_square_symmetry

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        if self.in_memory:
            graph = self.graphs[idx]
        else:
            graph_file = str(self.graph_files[idx].absolute())
            graph = load_graphs(graph_file)[0][0]
            if self.center_and_scale:
                graph.ndata["x"] = font_util.center_and_scale(graph.ndata["x"])
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
        return graph

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

    def get_dataloader(self, batch_size, shuffle):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            num_workers=helper.num_workers_platform(),
        )


def _collate_with_pointclouds(batch):
    graphs, pcs, graph_files = map(list, zip(*batch))
    bg = dgl.batch(graphs)
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return (
        bg,
        pcs,
        graph_files,
    )  # Return None for labels so that we can reuse same code in train/test scripts


class ABCDatasetWithPointclouds(Dataset):
    def __init__(
            self,
            bin_root_dir,
            npy_root_dir,
            split="train",
            size_percentage=None,
            in_memory=False,
            num_points=1024,
            apply_square_symmetry=0.0,
            center_and_scale=True,
    ):
        """
        ABC dataset with both solids and pointclouds
        :param bin_root_dir: Root path to the dataset of bin files
        :param npy_root_dir: Root path to the dataset of npy files
        :param split: Whether train, val, test or all (entire) set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param center_and_scale: Center and scale the UV grids
        """
        bin_path = pathlib.Path(bin_root_dir)
        npy_path = pathlib.Path(npy_root_dir)
        assert split in ("train", "val", "test", "all")

        pointcloud_files = list(npy_path.glob("*.npy"))
        # pointcloud_files = pointcloud_files[:1000]
        graph_files = list(bin_path.glob("*.bin"))
        # graph_files = graph_files[:1000]
        print(f"Found {len(graph_files)} {split} data.")
        print(f"Found pointcloud {len(pointcloud_files)} {split} data.")

        self.in_memory = in_memory
        self.center_and_scale = center_and_scale

        if size_percentage is not None:
            k = int(size_percentage * len(graph_files))
            graph_files = random.sample(graph_files, k)

        pc_hashmap = {}
        for file_name in pointcloud_files:
            # TODO: fix this weird extension and its handling
            query_name = file_name.name[:-4]  # remove .stl.npy extension
            pc_hashmap[query_name] = str(file_name)

        self.graph_files = []
        self.pc_files = []
        self.num_points = num_points

        for file_name in graph_files:
            query_name = file_name.name[:-4]  # remove .bin extension
            if query_name not in pc_hashmap:
                # print("Error: ", query_name)
                continue
            self.graph_files.append(str(file_name))
            self.pc_files.append(pc_hashmap[query_name])

        trainval_solids, test_solids, trainval_pc, test_pc = train_test_split(
            self.graph_files, self.pc_files, test_size=0.2, random_state=42,
        )
        train_solids, val_solids, train_pc, val_pc = train_test_split(
            trainval_solids, trainval_pc, test_size=0.25, random_state=84,
        )

        if split == "train":
            self.pc_files = train_pc
            self.graph_files = train_solids
        elif split == "val":
            self.pc_files = val_pc
            self.graph_files = val_solids
        elif split == "test":
            self.pc_files = test_pc
            self.graph_files = test_solids
        elif split == "all":
            pass
            # do nothing

        if self.in_memory:
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
            f"Loaded {len(self.graph_files)} face-adj graphs and {len(self.pc_files)} pointclouds"
        )

        self.apply_square_symmetry = apply_square_symmetry

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph_file = str(self.graph_files[idx])
        if self.in_memory:
            graph = self.graphs[idx]
            pc = self.pointclouds[idx][: self.num_points]
        else:
            graph = load_graphs(self.graph_files[idx])[0][0]
            pointcloud_file = self.pc_files[idx]
            pc = torch.tensor(np.load(pointcloud_file)[: self.num_points])
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
        return graph, pc, graph_file

    def get_dataloader(self, batch_size, shuffle):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_with_pointclouds,
            num_workers=helper.num_workers_platform(),
            drop_last=True,
        )


def _collate_only_pointclouds(batch):
    pcs = map(list, zip(*batch))
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return (
        pcs,
        None,
    )  # Return None for labels so that we can reuse same code in train/test scripts


class ABCDatasetPointclouds(Dataset):
    def __init__(
            self,
            npy_root_dir,
            split="train",
            size_percentage=None,
            in_memory=True,
            num_points=1024,
    ):
        """
        ABC dataset with pointclouds only
        :param npy_root_dir: Root path to the dataset of npy files
        :param split: Whether train, vall, test or all (entire) set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        """
        npy_path = pathlib.Path(npy_root_dir)
        assert split in ("train", "val", "test", "all")

        pointcloud_files = list(npy_path.glob("*.npy"))

        print(f"Found pointcloud {len(pointcloud_files)} {split} data.")

        self.in_memory = in_memory

        random.seed(1200)
        if split == "train":
            k = int(0.6 * len(pointcloud_files))
            pointcloud_files = random.sample(pointcloud_files, k)
        elif split == "val":
            k = int(0.2 * len(pointcloud_files))
            pointcloud_files = random.sample(pointcloud_files, k)
        elif split == "test":
            k = int(0.2 * len(pointcloud_files))
            pointcloud_files = random.sample(pointcloud_files, k)
        elif split == "all":
            pointcloud_files = pointcloud_files

        if size_percentage != None:
            k = int(size_percentage * len(pointcloud_files))
            pointcloud_files = random.sample(pointcloud_files, k)

        self.pc_files = pointcloud_files
        self.num_points = num_points

        if self.in_memory:
            print("Storing dataset in memory")
            print("Loading pointclouds...")
            self.pointclouds = [np.load(fn) for fn in self.pc_files]
        print(f"Loaded {len(self.pc_files)} pointclouds")

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        if self.in_memory:
            pc = self.pointclouds[idx][: self.num_points]
        else:
            pointcloud_file = self.pc_files[idx]
            pc = np.load(pointcloud_file)[: self.num_points]
        return pc

    def get_dataloader(self, batch_size, shuffle):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_only_pointclouds,
            num_workers=helper.num_workers_platform(),
        )


def _save_feature_to_csv(feat: torch.Tensor, filename):
    """
    Save loaded feature to csv file to visualize sampled points
    :param feat Features loaded from *.feat file of shape [#faces, #u, #v, 10]
    :param filename Output csv filename
    """
    assert len(feat.shape) == 4  # faces x #u x #v x 10
    pts = feat[:, :, :, :3].numpy().reshape((-1, 3))
    print(pts)
    mask = feat[:, :, :, 6].numpy().reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    np.savetxt(filename, pts, delimiter=",", header="x,y,z")


def _save_arr_to_csv(arr, filename):
    np.savetxt(filename, arr, delimiter=",", header="x,y,z")


if __name__ == "__main__":
    # split = "test"
    # test_dset = ABCDatasetWithPointclouds("c:\\users\\jayarap\\Downloads\\ABCFeat\\feat",
    #                                       "c:\\users\\jayarap\\Downloads\\ABCFeat\\pointclouds", split=split)
    # print("Creating dataloader")
    # shuffle = True  #(split != "test")
    # dataloader = DataLoader(test_dset, batch_size=1, shuffle=shuffle,
    #                         num_workers=0, collate_fn=collate_with_pointclouds)
    # for i, (graph, pc) in enumerate(dataloader):
    #     feat = graph.ndata['x']  # .permute(0, 3, 1, 2)
    #     print(feat)
    #     _save_feature_to_csv(feat, "c:\\users\\jayarap\\abctest_graph.csv")
    #     _save_arr_to_csv(pc[0], "c:\\users\\jayarap\\abctest_pc.csv")
    #     if i == 0:
    #         break
    # filename = "c:\\users\\jayarap\\downloads\\abcfeat\\abctest.bin"
    # # filename = "c:\\users\\jayarap\\downloads\\abcfeat\\feat\\- Part 1-ft4sj041azoij61gik0d.bin"
    # g = load_graphs(filename)[0][0]
    # _save_feature_to_csv(
    #     g.ndata["x"], "c:\\users\\jayarap\\downloads\\abcfeat\\abctest.csv"
    # )

    dset = ABCDatasetWithPointclouds(bin_root_dir='/home/ubuntu/abc/bin',
                                     npy_root_dir='/home/ubuntu/abc/pointclouds',
                                     split='all',
                                     center_and_scale=False)
    graph_files = []
    num_nodes = []
    for graph, _, file in tqdm(dset):
        num_nodes.append(graph.number_of_nodes())
        graph_files.append(file)

    df = pd.DataFrame({
        'num_nodes': num_nodes,
        'graph_file': graph_files
    })
    df.to_csv('abc_num_nodes.csv', sep=',', index=False)
