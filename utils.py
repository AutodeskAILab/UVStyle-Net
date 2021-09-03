import os
import pickle
import sys
from collections import OrderedDict

import PIL
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances


class Grams(object):
    def __init__(self, grams_root, label_map=None):
        self.data_root = grams_root
        self.file_names = sorted(list(filter(lambda file_name: file_name[-4:] == '.npy', os.listdir(grams_root))))
        self.layer_names = list(map(lambda filename: '_'.join(filename.split('_')[:-1]), self.file_names))
        self._label_map = label_map
        self._grams = None
        self._graph_files = None
        self._labels = None

    @property
    def graph_files(self):
        if self._graph_files is None:
            self._graph_files = pd.read_csv(os.path.join(self.data_root, 'graph_files.txt'), header=None)[0].tolist()
        return self._graph_files

    @property
    def labels(self):
        if self._labels is None:
            names = map(lambda file: file[2:-10], self.graph_files)
            if self._label_map is None:
                df = pd.DataFrame(names, columns=['name'])
                self._labels = df['name'].astype('category').cat.codes.values
            else:
                self._labels = list(map(self._label_map, names))
        return self._labels

    @property
    def grams(self):
        if self._grams is None:
            layer_paths = map(lambda layer_name: os.path.join(self.data_root, layer_name), self.file_names)
            self._grams = list(map(np.load, layer_paths))
        return self._grams

    def __getitem__(self, item):
        return self.grams[item]

    def __len__(self):
        return len(self.grams)


class ImageLoader(object):
    def __init__(self, grams_root, img_root, img_type='png', black_to_white=False):
        self.data_root = grams_root
        self.img_root = img_root
        self.img_type = img_type
        self.black_to_white = black_to_white
        graph_files = pd.read_csv(os.path.join(self.data_root, 'graph_files.txt'), header=None)[0].tolist()
        img_files = map(lambda file_name: file_name[:-3] + self.img_type, graph_files)
        self.img_paths = np.array(list(map(lambda img_name: os.path.join(self.img_root, img_name), img_files)))

    def __getitem__(self, item):
        img_paths = self.img_paths[item].flatten()
        pngs = map(self.open_image, img_paths)
        return list(map(lambda png: png.convert('RGB'), pngs))

    def open_image(self, img_path):
        try:
            img = PIL.Image.open(img_path)
            if self.black_to_white:
                arr = np.array(img.convert('RGB'))

                black_x, black_y = np.where((arr == [0, 0, 0]).all(axis=-1))
                green_x, green_y = np.where((arr == [0, 255, 0]).all(axis=-1))

                arr = (arr + 50).astype(np.uint8)

                arr[black_x, black_y, :] = [255, 255, 255]
                arr[green_x, green_y, :] = [0, 0, 0]

                img = PIL.Image.fromarray(arr)
        except FileNotFoundError as e:
            print(f'WARNING cannot find {img_path}, using blank image - {e}', file=sys.stderr)
            img = PIL.Image.new(mode='P', size=(512, 512), color=(255, 255, 255))
        return img


def gram_loss(grams, id_a, id_b, weights, metric='cosine'):
    loss = 0
    for i in range(len(grams)):
        a = grams[i][id_a]
        b = grams[i][id_b]
        if metric == 'cosine':
            loss += weights[i] * paired_cosine_distances(a[None, :], b[None, :])
        elif metric == 'euclidean':
            loss += weights[i] * paired_euclidean_distances(a[None, :], b[None, :])
    return id_b, loss


def top_k_neighbors(X, weights, queries, k, metric='cosine'):
    all_neighbors = []
    all_distances = []
    for query in queries:
        q = X[query]
        if metric == 'cosine':
            layerwise_distances = 1 - torch.cosine_similarity(q[None, :, :], X, dim=-1)
        elif metric == 'euclidean':
            layerwise_distances = torch.norm(q[None, :, :] - X, dim=-1)
        else:
            raise Exception(f'{metric} not found - use "cosine" or "euclidean"')
        weighted = layerwise_distances * weights[None, :]
        distances = weighted.sum(-1)  # type: torch.Tensor
        neighbors = torch.argsort(distances)[:k]
        all_neighbors.append(neighbors)
        all_distances.append(distances[neighbors])
    all_neighbors = torch.stack(all_neighbors).detach().cpu().numpy()
    all_distances = torch.stack(all_distances).detach().cpu().numpy()
    return all_neighbors, all_distances


def pad_grams(X):
    grams_0 = torch.zeros(len(X[0]), 70)
    grams_0[:, :X[0].shape[-1]] = torch.tensor(X[0].copy())
    grams_padded = torch.stack([grams_0] + [torch.tensor(gram.copy()) for gram in X[1:]])
    X = grams_padded.permute(1, 0, 2)  # shape: N x 7 x 70
    return X


def get_pca_70(grams, cache_file, verbose=False):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            pca_70 = pickle.load(file)
    else:
        pca_70 = {}
        for layer_name, gram in zip(grams.layer_names, grams):
            if verbose:
                print(f'pca: {layer_name}...')
            if gram.shape[-1] <= 70:
                pca_70[layer_name] = gram
            else:
                pca = PCA(n_components=70)
                pca_70[layer_name] = pca.fit_transform(gram)
        pca_70 = OrderedDict(sorted(pca_70.items()))
        with open(cache_file, 'wb') as file:
            pickle.dump(pca_70, file)

    return pca_70


def dataset_selector(project_root):
    dataset_name = st.sidebar.selectbox(label='Dataset',
                                        options=os.listdir(os.path.join(project_root, 'data')))
    data_root = os.path.join(project_root, 'data', dataset_name)

    grams_name = st.sidebar.selectbox(label='Grams',
                                      options=os.listdir(os.path.join(data_root, 'grams')))

    grams_root = os.path.join(data_root, 'grams', grams_name)
    return data_root, dataset_name, grams_root, grams_name


def plot(grid_array, query_idx, img_size, k, distances=None, font_size=24):
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.imshow(grid_array)

    ax.set_xticks(np.arange(k) * (img_size + 2) + (img_size / 2))
    ax.set_xticklabels(['Q'] + list(map(str, np.arange(1, k))), Fontsize=font_size)

    if query_idx is not None and len(query_idx) > 0:
        ax.set_yticks(np.arange(len(query_idx)) * (img_size + 2) + (img_size / 2))
        ax.set_yticklabels(query_idx if query_idx is not None else None, Fontsize=font_size)
    else:
        ax.set_yticks([])

    if distances is not None:
        x = np.arange(k)
        y = np.arange(len(distances))
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        text = list(map(lambda d: f'{d:.2f}', distances.tolist()))

        space = img_size
        for t, pos in zip(text, xy):
            x, y = pos
            ax.annotate(t, ((x * space + .6 * space), (y * space + .9 * space)), color='red', Fontsize=font_size)

    fig.set_size_inches(21, 15)
    fig.tight_layout()
    return fig


def warn_and_get_pca70(file_dir, dataset_name, grams_name, grams):
    pca_70 = None
    os.makedirs(os.path.join(file_dir, 'cache'), exist_ok=True)
    cache_file = os.path.join(file_dir, 'cache', f'{dataset_name}-{grams_name}-pca_70')
    if not os.path.exists(cache_file):
        st.write('To visualize the query results, you will first need to perform PCA on the Gram matrices.')
        st.write('You will need to do this only once.')
        st.write('For SolidLETTERS this should take less than a minute, and require no more then 1GB memory.')
        st.write('For ABC this will require approx. 36GB memory, and will take a few minutes.')
        if st.button(label='Start'):
            pca_70 = get_pca_70(grams=grams,
                                cache_file=cache_file,
                                verbose=True)
    else:
        pca_70 = get_pca_70(grams=grams,
                            cache_file=cache_file,
                            verbose=True)
    return pca_70
