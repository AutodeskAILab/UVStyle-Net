import math
import os
import pathlib
import platform
import random

import PIL
import dgl
import numpy as np
import torch
import torchvision
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset

from font_util import valid_font


def simclr_collate(batch):
    g1_g2s, labels, meta, images, graph_files = map(list, zip(*batch))
    g1s, g2s = zip(*g1_g2s)
    labels = torch.stack(labels)
    bg1 = dgl.batch(g1s)
    bg2 = dgl.batch(g2s)
    return (bg1, bg2), labels


def original_collate(batch):
    graphs, labels, files = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    return bg, labels, files


def my_collate(batch):
    graphs, labels, meta, images, graph_files = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    metas = torch.stack(meta)
    images = torch.stack(images)
    return bg, labels, metas, images, graph_files


def collate_single_letter(batch):
    graphs, labels, graph_files = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    return bg, labels, graph_files


def _make_dgl_graph(xs, edges, label):
    g = dgl.DGLGraph()
    g.add_nodes(xs.shape[0])
    g.ndata['x'] = torch.tensor(xs).float()
    g.add_edges(edges[:, 0], edges[:, 1])
    g.ndata['y'] = (torch.ones(xs.shape[0]) * label).long()
    return g


class SolidLETTERS(Dataset):
    def __init__(self, root_dir, split="train", size_percentage=None, in_memory=False, apply_square_symmetry=0.0,
                 split_suffix="", transform=None, crop_func=None, image_dir=None):
        """
        Load and create the SolidMNIST dataset
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param split_suffix: Suffix for the split directory to use
        """
        self.transform = transform
        self.crop_func = crop_func
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")
        if split == "train" or split == "val":
            subfolder = "train"
        else:
            subfolder = "test"

        path /= subfolder + split_suffix
        self.graph_files = list(x for x in path.glob("*.bin") if valid_font(x))
        print("Found {} {} data.".format(len(self.graph_files), split))
        self.labels = []

        self.in_memory = in_memory

        if split == "train":
            k = int(.8 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "val":
            k = int(.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        for fn in self.graph_files:
            # the first character of filename must be the alphabet
            self.labels.append(self.char_to_label(fn.stem[0]))
        self.num_classes = len(set(self.labels))
        if platform.system() == "Windows" or in_memory:
            print("Windows OS detected, storing dataset in memory")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
        print("Done loading {} and data {} classes".format(len(self.graph_files), self.num_classes))
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
        if self.apply_square_symmetry > 0.0:
            prob_r = random.uniform(0.0, 1.0)
            if prob_r < self.apply_square_symmetry:
                graph.ndata['x'] = graph.ndata['x'].transpose(1, 2)
            prob_u = random.uniform(0.0, 1.0)
            if prob_u < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[1])
            prob_v = random.uniform(0.0, 1.0)
            if prob_v < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[2])

        x = graph.ndata['x']
        x = corner_align(x)
        graph.ndata['x'] = x
        # get extra label info
        stem = self.graph_files[idx].stem.lower()
        name = stem[2:-6]
        upper = stem[-5:] == 'upper'
        light = name.find('light') > -1
        bold = name.find('bold') > -1
        script = name.find('script') > -1
        sans = name.find('sans') > -1
        letter = self.char_to_label(stem[0])
        meta = torch.tensor([upper, light, bold, script, sans, letter], dtype=torch.long)
        if self.transform:
            graph_file = str(self.graph_files[idx].absolute())
            graph2 = load_graphs(graph_file)[0][0]  # type: dgl.DGLGraph

            x1 = graph.ndata['x']
            x1 = self.transform(x1)
            graph.ndata['x'] = x1

            x2 = graph.ndata['x']
            x2 = self.transform(x2)
            graph2.ndata['x'] = x2

            graph = (graph, graph2)

        if self.crop_func:
            graph = map(self.crop_func, graph)

        return graph, torch.tensor([self.labels[idx]]).long(), meta, torch.zeros(1), self.graph_files[idx].stem + '.bin'

    def feature_indices(self):
        """
        Returns a dictionary of mappings from the name of features to the channels containing them
        """
        return {"xyz": (0, 1, 2), "normals": (3, 4, 5), "mask": (6,), "E": (7,), "F": (8,), "G": (9,)}


class SolidLETTERSSubset(Dataset):
    def __init__(self, root_dir, split="train", size_percentage=None, in_memory=False, apply_square_symmetry=0.0,
                 split_suffix="", image_dir=None,
                 random_rotation=None):
        """
        Load and create the SolidMNIST dataset
        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        :param size_percentage: Percentage of data to load per category
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_square_symmetry: Probability of randomly applying a square symmetry transform on the input surface grid (default: 0.0)
        :param split_suffix: Suffix for the split directory to use
        :param random_rotation: 'x' or 'z' or None
        """
        self.image_dir = image_dir
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")
        if split == "train" or split == "val":
            subfolder = "train"
        else:
            subfolder = "test"

        path /= subfolder + split_suffix
        self.graph_files = list(x for x in path.glob("*.bin") if valid_font(x))
        self.fonts = {
            'turret road': 0,
            'zhi mang xing': 1,
            'abhaya libre': 2,
            'seaweed script': 3
        }

        def matching_font(file_name):
            font_name = file_name.stem[2:-6]
            return font_name.lower() in self.fonts.keys()

        self.graph_files = list(filter(matching_font, self.graph_files))
        print("Found {} {} data.".format(len(self.graph_files), split))
        self.labels = []

        self.in_memory = in_memory

        if split == "train":
            k = int(.8 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "val":
            k = int(.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        for fn in self.graph_files:
            # the first character of filename must be the alphabet
            self.labels.append(self.char_to_label(fn.stem[0]))
        # self.num_classes = len(set(self.labels))
        self.num_classes = 26
        if platform.system() == "Windows" or in_memory:
            print("Windows OS detected, storing dataset in memory")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
        print("Done loading {} and data {} classes".format(len(self.graph_files), self.num_classes))
        self.apply_square_symmetry = apply_square_symmetry
        self.random_rotation = random_rotation

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
        if self.apply_square_symmetry > 0.0:
            prob_r = random.uniform(0.0, 1.0)
            if prob_r < self.apply_square_symmetry:
                graph.ndata['x'] = graph.ndata['x'].transpose(1, 2)
            prob_u = random.uniform(0.0, 1.0)
            if prob_u < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[1])
            prob_v = random.uniform(0.0, 1.0)
            if prob_v < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[2])

        if self.random_rotation is not None:
            graph.ndata['x'] = random_rotate(graph.ndata['x'], self.random_rotation)

        x = graph.ndata['x']
        x = corner_align(x)
        graph.ndata['x'] = x

        # get extra label info
        stem = self.graph_files[idx].stem.lower()
        name = stem[2:-6]
        upper = stem[-5:] == 'upper'
        light = name.find('light') > -1
        bold = name.find('bold') > -1
        script = name.find('script') > -1
        sans = name.find('sans') > -1
        font = self.fonts[name]
        letter = self.char_to_label(stem[0])

        # get image
        image_path = os.path.join(self.image_dir, self.graph_files[idx].stem + '.png')
        try:
            image = PIL.Image.open(image_path)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([480, 480]),
                torchvision.transforms.ToTensor()
            ])
            image = transform(image)
        except Exception as e:
            print(f'Warning could not load image {image_path} - {e}')
            image = torch.ones([3, 480, 480])

        meta = torch.tensor([upper, light, bold, script, sans, font, letter], dtype=torch.long)
        return graph, torch.tensor([self.labels[idx]]).long(), meta, image, str(self.graph_files[idx].name)

    def feature_indices(self):
        """
        Returns a dictionary of mappings from the name of features to the channels containing them
        """
        return {"xyz": (0, 1, 2), "normals": (3, 4, 5), "mask": (6,), "E": (7,), "F": (8,), "G": (9,)}


def collate_with_pointclouds(batch):
    graphs, pcs, labels, graph_files = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    pcs = torch.from_numpy(np.stack(pcs)).type(torch.FloatTensor)
    return bg, pcs, labels, graph_files


class SolidMNISTWithPointclouds(Dataset):
    def __init__(self, bin_root_dir, npy_root_dir, split="train", shape_type=None, size_percentage=None, in_memory=True,
                 num_points=1024,
                 apply_square_symmetry=0.0, split_suffix=""):
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
        pointcloud_files = list([x for x in npy_path.glob("*.npy") if valid_font(x)])
        graph_files = list([x for x in bin_path.glob("*.bin") if valid_font(x)])
        print("Found {} {} data.".format(len(graph_files), subfolder))
        print("Found pointcloud {} {} data.".format(len(pointcloud_files), subfolder))

        self.in_memory = in_memory

        if split == "train":
            k = int(.8 * len(graph_files))
            graph_files = random.sample(graph_files, k)
        elif split == "val":
            k = int(.2 * len(graph_files))
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
            self.pointclouds = [np.load(fn) for fn in self.pc_files]
        print("Loaded {} face-adj graphs and {} pc files from {} classes".format(len(self.graph_files),
                                                                                 len(self.pc_files), self.num_classes))

        self.apply_square_symmetry = apply_square_symmetry

    def __len__(self):
        return len(self.graph_files)

    def char_to_label(self, char):
        return ord(char.lower()) - 97

    def __getitem__(self, idx):
        if platform.system() == "Windows" or self.in_memory:
            graph = self.graphs[idx]
            pc = self.pointclouds[idx][:self.num_points]
        else:
            graph_file = str(self.graph_files[idx])
            graph = load_graphs(self.graph_files[idx])[0][0]
            pointcloud_file = self.pc_files[idx]
            pc = np.load(pointcloud_file)[:self.num_points]  # ['arr_0']
        x = graph.ndata['x']
        x = corner_align(x)
        graph.ndata['x'] = x
        if self.apply_square_symmetry > 0.0:
            prob_r = random.uniform(0.0, 1.0)
            if prob_r < self.apply_square_symmetry:
                graph.ndata['x'] = graph.ndata['x'].transpose(1, 2)
            prob_u = random.uniform(0.0, 1.0)
            if prob_u < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[1])
            prob_v = random.uniform(0.0, 1.0)
            if prob_v < self.apply_square_symmetry:
                graph.ndata['x'] = torch.flip(graph.ndata['x'], dims=[2])
        label = torch.tensor([self.labels[idx]]).long()
        return graph, pc, label, pathlib.Path(self.graph_files[idx]).name


class SolidMNISTWithPointcloudsFontSubset(SolidMNISTWithPointclouds):
    def __init__(self, bin_root_dir, npy_root_dir, split='test'):
        super(SolidMNISTWithPointcloudsFontSubset, self).__init__(bin_root_dir, npy_root_dir, split)
        self.fonts = {
            'turret road': 0,
            'zhi mang xing': 1,
            'abhaya libre': 2,
            'seaweed script': 3
        }

        def matching_font(file_name):
            font_name = file_name.stem[2:-6]
            return font_name.lower() in self.fonts.keys()

        self.graph_files = list(map(pathlib.Path, self.graph_files))

        self.graph_files, self.pointclouds, self.labels = zip(*list(
            filter(lambda x: matching_font(x[0]), zip(self.graph_files, self.pointclouds, self.labels))
        ))


def _save_feature_to_csv(feat: torch.Tensor, filename):
    """
    Save loaded feature to csv file to visualize sampled points
    :param feat Features loaded from *.feat file of shape [#faces, #u, #v, 10]
    :param filename Output csv filename
    """
    assert len(feat.shape) == 4  # faces x #u x #v x 10
    pts = feat[:, :, :, :3].numpy().reshape((-1, 3))
    mask = feat[:, :, :, 6].numpy().reshape(-1)
    point_indices_inside_faces = (mask == 1)
    pts = pts[point_indices_inside_faces, :]
    np.savetxt(filename, pts, delimiter=",", header="x,y,z")


def _category_count(path):
    import matplotlib.pyplot as plt
    import pathlib
    import string
    import numpy as np

    alphabets = list(string.ascii_lowercase)
    path = pathlib.Path(path)
    count = np.zeros(len(alphabets))
    max_count = 0
    for i, alphabet in enumerate(alphabets):
        files = list(path.glob(f'[{alphabet}]*.bin'))
        count[i] = len(files)
        max_count = max(max_count, count[i])
    count /= max_count
    plt.bar(np.arange(len(alphabets)), count, tick_label=alphabets)
    plt.show()


def bounding_box(input: torch.Tensor):
    x = input[:, :, :, 0]
    y = input[:, :, :, 1]
    z = input[:, :, :, 2]
    box = [
        [x.min(), y.min(), z.min()],
        [x.max(), y.max(), z.max()]
    ]
    return torch.tensor(box)


def corner_align(input: torch.Tensor):
    x = input[:, :, :, 0]
    y = input[:, :, :, 1]
    z = input[:, :, :, 2]
    min_x, _ = x.flatten().min(dim=-1)
    min_y, _ = y.flatten().min(dim=-1)
    min_z, _ = z.flatten().min(dim=-1)

    mins = torch.stack([min_x, min_y, min_z], dim=-1)
    out = input[:, :, :, :3] - mins[None, None, None, :]
    return torch.cat([out, input[:, :, :, 3:]], dim=-1)


def rotate_x(xyz: torch.Tensor, theta: torch.Tensor):
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    r_x = torch.tensor([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    return torch.matmul(xyz, r_x)


def rotate_z(xyz: torch.Tensor, theta: torch.Tensor):
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    r_z = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return torch.matmul(xyz, r_z)


def random_rotate(x: torch.Tensor, axis: str):
    theta = torch.rand(1) * 2 * math.pi
    xyz = x[:, :, :, :3]
    normals = x[:, :, :, 3:6]
    rest = x[:, :, :, 6:]
    if axis == 'x':
        xyz = rotate_x(xyz, theta)
        normals = rotate_x(normals, theta)
    elif axis == 'z':
        xyz = rotate_z(xyz, theta)
        normals = rotate_z(normals, theta)
    else:
        raise NotImplementedError('random rotation should be \'x\' or \'z\'')
    rotated = torch.cat([xyz, normals, rest], dim=-1)
    aligned = corner_align(rotated)
    return aligned


def identity_transform(x):
    return x
