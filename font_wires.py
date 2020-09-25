import os.path as osp
import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader
import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
import platform
import random
from font_util import valid_font
from PIL import Image


def collate_graphs(batch):
    graphs, labels = map(list, zip(*batch))
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    return bg, labels

def collate_graphs_with_images(batch):
    graphs, images, labels = map(list, zip(*batch))
    #print(images[0].shape)
    labels = torch.stack(labels)
    bg = dgl.batch(graphs)
    images = torch.from_numpy(np.stack(images)).type(torch.FloatTensor)
    return bg, images, labels

def _make_dgl_graph(xs, edges, label):
    g = dgl.DGLGraph()
    g.add_nodes(xs.shape[0])
    g.ndata['x'] = torch.tensor(xs).float()
    g.add_edges(edges[:, 0], edges[:, 1])
    g.ndata['y'] = (torch.ones(xs.shape[0]) * label).long()
    return g


class FontWires(Dataset):
    def __init__(self, root_dir, split="train", size_percentage=None, in_memory=True, apply_line_symmetry=0.0):
        """
        Load and create the FontWires dataset
        :param root_dir: Root path to the dataset
        :param train: Whether train or test set
        :param size_percentage: Percentage of data to load per category. Given in range [0, 1].
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_line_symmetry: Probability of applying a line symmetry transform on the curve grid
        """
        assert split in ("train", "val", "test")
        self.in_memory = in_memory
        path = pathlib.Path(root_dir)
        if split == "train" or split == "val":   
            subfolder = "train"
        else:
            subfolder = "test"
        path /= subfolder
        self.graph_files = list([x for x in path.glob("*.bin") if valid_font(x)])
        print("Found {} {} data.".format(len(self.graph_files), subfolder))
        self.labels = []
        
        if split == "train":
            k = int(0.8 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "val":
            k = int(0.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
            
        for fn in self.graph_files:
            # the first character of filename must be the alphabet
            self.labels.append(self.char_to_label(fn.stem[0]))
        self.num_classes = len(set(self.labels))
        if platform.system() == "Windows" or self.in_memory:
            print("Storing dataset in memory...")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.graph_files]
        print("Done loading {} classes".format(self.num_classes))
        self.apply_line_symmetry = apply_line_symmetry

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
        if self.apply_line_symmetry > 0.0:
            prob = random.uniform(0.0, 1.0)
            if prob < self.apply_line_symmetry:
                graph.ndata['x'] = graph.ndata['x'].flip(1)
        return graph, torch.tensor([self.labels[idx]]).long()

    def feature_indices(self):
        """
        Returns a dictionary of mappings from the name of features to the channels containing them
        """
        return {"xyz": (0, 1, 2), "normals": (3, 4, 5), "E (ru.ru)": (6,)}


    
class FontWiresWithImages(Dataset):
    def __init__(self, root_dir, split="train", size_percentage=None, in_memory=True, shape_type= None, apply_line_symmetry=0.0):
        """
        Load and create the FontWires dataset
        :param root_dir: Root path to the dataset
        :param train: Whether train or test set
        :param size_percentage: Percentage of data to load per category. Given in range [0, 1].
        :param in_memory: Whether to keep the entire dataset in memory (This is always done in Windows)
        :param apply_line_symmetry: Probability of applying a line symmetry transform on the curve grid
        """
        assert split in ("train", "val", "test")
        self.in_memory = in_memory
        path = pathlib.Path(root_dir)
        if split == "train" or split == "val":   
            subfolder = "train"
        else:
            subfolder = "test"
        path /= subfolder
        self.graph_files = list([x for x in path.glob("*.bin") if valid_font(x)])
        self.images_files = list([x for x in path.glob("*.png") if valid_font(x)])
        
        print("Found {} {} {} data.".format(len(self.graph_files), len(self.images_files), subfolder))
        self.labels = []
        
        if split == "train":
            k = int(0.8 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        elif split == "val":
            k = int(0.2 * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)

        if size_percentage is not None:
            k = int(size_percentage * len(self.graph_files))
            self.graph_files = random.sample(self.graph_files, k)
        
        self.img_hashmap = {}
        for file_name in self.images_files:
            query_name = str(file_name).split(".")[0]
            self.img_hashmap[query_name]  = str(file_name)
        
        self.final_graph_files = []
        self.final_image_files = []
        self.labels = []
        
        for file_name in self.graph_files:
            # the first character of filename must be the alphabet
            query_name = str(file_name).split(".")[0]
            if shape_type is not None and shape_type not in query_name:
                continue
          
            if query_name not in self.img_hashmap:
                #print("Error: ", query_name)
                continue 
            self.final_graph_files.append(str(file_name))
            self.final_image_files.append(self.img_hashmap[query_name])
            self.labels.append(self.char_to_label(file_name.stem[0]))
            
            
        self.num_classes = len(set(self.labels))
        self.images = []
        if platform.system() == "Windows" or self.in_memory:
            print("Storing dataset in memory...")
            self.graphs = [load_graphs(str(fn))[0][0] for fn in self.final_graph_files]
            for fn in self.final_image_files:
                image = Image.open(fn)
                image = image.resize((64, 64), Image.ANTIALIAS).convert('L') 
                image = np.expand_dims(np.asarray(image)/255.0, axis=-1)
                self.images.append(image)
            #self.images = [np.load(fn) for fn in self.final_image_files]
        
        self.graph_files = self.final_graph_files  
        self.image_files = self.final_image_files
        
        print("Loaded {} face-adj graphs and {} image files from {} classes".format(len(self.graph_files), len(self.image_files), self.num_classes))
        self.apply_line_symmetry = apply_line_symmetry

    def char_to_label(self, char):
        return ord(char.lower()) - 97

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        if platform.system() == "Windows" or self.in_memory:
            graph = self.graphs[idx]
            image = self.images[idx]
        else:
            graph_file = str(self.graph_files[idx].absolute())
            graph = load_graphs(graph_file)[0][0]
            image = Image.open(self.image_files[idx])
            image = image.resize((64, 64), Image.ANTIALIAS).convert('L') 
            image = np.expand_dims(np.asarray(image)/255.0, axis=-1)
            
        if self.apply_line_symmetry > 0.0:
            prob = random.uniform(0.0, 1.0)
            if prob < self.apply_line_symmetry:
                graph.ndata['x'] = graph.ndata['x'].flip(1)
        return graph, image, torch.tensor([self.labels[idx]]).long()

    def feature_indices(self):
        """
        Returns a dictionary of mappings from the name of features to the channels containing them
        """
        return {"xyz": (0, 1, 2), "normals": (3, 4, 5), "E (ru.ru)": (6,)}
    
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


#if __name__ == "__main__":
    # dataloader = DataLoader(dset, batch_size=2, shuffle=True,
    #                         num_workers=0, collate_fn=my_collate)
    # data, labels = next(iter(dataloader))
    # feat = data.ndata['x']  # .permute(0, 3, 1, 2)
    # labels = data.ndata['y']
    # print(feat.shape)
    # print(dset.num_classes)
    # _save_feature_to_csv(feat, "c:\\users\\jayarap\\test.csv")
