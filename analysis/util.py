import os
import pickle
import sys
from collections import OrderedDict

import PIL.Image
import numpy as np
import pandas as pd
import torch
import torchvision
import streamlit as st
from plotly import graph_objs as go, express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, paired_cosine_distances
from sklearn.neighbors._kd_tree import KDTree
import matplotlib.pyplot as plt


class Grams(object):
    def __init__(self, data_root, label_map=None):
        self.data_root = data_root
        self.file_names = sorted(list(filter(lambda file_name: file_name[-4:] == '.npy', os.listdir(data_root))))
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


class OnTheFlyImages(object):
    def __init__(self, data_root, img_root, img_type='png'):
        self.data_root = data_root
        self.img_root = img_root
        self.img_type = img_type
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
        except FileNotFoundError as e:
            print(f'WARNING cannot find {img_path}, using blank image - {e}', file=sys.stderr)
            img = PIL.Image.new(mode='P', size=(512, 512), color=(255, 255, 255))
        return img


def __len__(self):
    return


class Images(object):
    def __init__(self, data_root, img_root, cache_file, img_type='png'):
        self.data_root = data_root
        self.img_root = img_root
        self.cache_file = cache_file + '.pickle'
        self.img_type = img_type
        self._imgs = None

    @property
    def imgs(self):
        if self._imgs is None:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as file:
                    self._imgs = pickle.load(file)
            else:
                graph_files = pd.read_csv(os.path.join(self.data_root, 'graph_files.txt'), header=None)[0].tolist()
                img_files = map(lambda file_name: file_name[:-3] + self.img_type, graph_files)
                img_paths = map(lambda img_name: os.path.join(self.img_root, img_name), img_files)
                pngs = map(self.open_image, img_paths)
                self._imgs = list(map(lambda png: png.convert('RGB'), pngs))
                with open(self.cache_file, 'wb') as file:
                    pickle.dump(self._imgs, file)
        return self._imgs

    def __getitem__(self, item):
        return self.imgs[item]

    def __len__(self):
        return len(self.imgs)

    def open_image(self, img_path):
        try:
            img = PIL.Image.open(img_path)
        except FileNotFoundError as e:
            print(f'WARNING cannot find {img_path}, using blank image - {e}', file=sys.stderr)
            img = PIL.Image.new(mode='P', size=(512, 512), color=(255, 255, 255))
        return img


def get_pca_3_70(grams, cache_file, verbose=False):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            pca_3, pca_70 = pickle.load(file)
    else:
        pca_3 = {}
        for layer_name, gram in zip(grams.layer_names, grams):
            pca = PCA(n_components=3)
            pca_3[layer_name] = pca.fit_transform(gram)
        pca_3 = OrderedDict(sorted(pca_3.items()))

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
            pickle.dump((pca_3, pca_70), file)

    return pca_3, pca_70


class KNNGrid(object):
    def __init__(self, kd_tree, images, resize=256):
        self.images = images
        self.kd_tree = kd_tree
        self.img_size = resize

    def _get_image(self, queries, k=5, label_fn=None, thickness=10):
        match_idx = self.kd_tree.query(queries, k=k, return_distance=False)
        print(match_idx)
        errors = np.zeros_like(match_idx)
        if label_fn is not None:
            labels = label_fn(match_idx)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] != labels[i, 0]:
                        errors[i, j] = 1
        print(errors)
        imgs = self.images[match_idx.flatten()]
        t = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.img_size, self.img_size)),
            torchvision.transforms.ToTensor()
        ])

        def to_tensor(img, error):
            img_tensor = t(img)  # type: torch.Tensor
            if error:
                color = torch.tensor([1., 0., 0.])
                x = img_tensor.permute(1, 2, 0)
                x[:thickness, :, :] = color
                x[-thickness:, :, :] = color
                x[:, :thickness, :] = color
                x[:, -thickness:, :] = color
                return x.permute(2, 0, 1)
            return img_tensor

        img_tensors = [to_tensor(img, err) for img, err in zip(imgs, errors.flatten())]
        return torchvision.utils.make_grid(img_tensors, nrow=k).permute((1, 2, 0))

    def get_plotly(self, queries, k=5, query_idx=None):
        grid_array, match_idx = self._get_image(queries, k=k)
        fig = go.Figure(px.imshow(grid_array))  # type: go.Figure
        fig.update_layout(yaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(queries)) * (self.img_size + 2) + (self.img_size / 2),
            ticktext=query_idx if query_idx is not None else None
        ),
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(k) * (self.img_size + 2) + (self.img_size / 2),
                ticktext=['Q'] + list(map(str, np.arange(1, k)))
            ))
        return fig


class KNNGridWithDistances(object):
    def __init__(self, kd_tree, images, raw_emb, resize=1024):
        self.images = images
        self.kd_tree = kd_tree
        self.img_size = resize
        self.raw_emb = raw_emb

    def _get_image(self, queries, k=5):
        match_idx = self.kd_tree.query(queries, k=k, return_distance=False)
        embs = self.raw_emb[match_idx]
        distances = np.stack([
            paired_cosine_distances(embs[:, 0, :], embs[:, i, :]) for i in range(embs.shape[1])
        ], axis=-1)
        print(match_idx)
        imgs = self.images[match_idx.flatten()]
        t = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.img_size, self.img_size)),
            torchvision.transforms.ToTensor()
        ])
        img_tensors = list(map(t, imgs))
        return torchvision.utils.make_grid(img_tensors, nrow=k).permute((1, 2, 0)), distances

    def get_plotly(self, queries, k=5, query_idx=None, with_distance=True):
        grid_array, distances = self._get_image(queries, k=k)
        fig = go.Figure(px.imshow(grid_array))  # type: go.Figure
        fig.update_layout(yaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(queries)) * (self.img_size + 2) + (self.img_size / 2),
            ticktext=query_idx if query_idx is not None else None
        ),
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(k) * (self.img_size + 2) + (self.img_size / 2),
                ticktext=['Q'] + list(map(str, np.arange(1, k)))
            ))

        x = np.arange(k)
        y = np.arange(distances.shape[0])
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        text = list(map(lambda d: f'{d:.2f}', distances.flatten().tolist()))
        fig.add_trace(go.Scatter(x=xy[:, 0] * 128 + 64,
                                 y=xy[:, 1] * 128 + 120,
                                 text=text,
                                 mode='text',
                                 textfont={'size': 16, 'color': "DarkOrange"}))
        return fig

    def get_matplotlib(self, queries, k=5, query_idx=None, with_distance=True):
        grid_array, distances = self._get_image(queries, k=k)
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        ax.imshow(grid_array)

        ax.set_xticks(np.arange(k) * (self.img_size + 2) + (self.img_size / 2))
        ax.set_xticklabels(['Q'] + list(map(str, np.arange(1, k))), Fontsize=24)

        ax.set_yticks(np.arange(len(queries)) * (self.img_size + 2) + (self.img_size / 2))
        ax.set_yticklabels(query_idx if query_idx is not None else None, Fontsize=24)

        if with_distance:
            x = np.arange(k)
            y = np.arange(distances.shape[0])
            xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
            text = list(map(lambda d: f'{d:.2f}', distances.flatten().tolist()))

            space = self.img_size
            for t, pos in zip(text, xy):
                x, y = pos
                ax.annotate(t, ((x * space + .6 * space), (y * space + .9 * space)), color='orange', Fontsize=20)

        fig.set_size_inches(21, 15)
        fig.tight_layout()
        return fig


class StImageSelectorTextBoxes(object):
    def __init__(self, imgs, id_map=None):
        self.imgs = imgs
        st.sidebar.text('Example ids (space separated)')
        pos_text = st.sidebar.text_area('positives:')
        neg_text = st.sidebar.text_area('negatives:')
        self.positives = list(map(int, pos_text.split(' ')))
        self.negatives = list(map(int, neg_text.split(' ')))

        if id_map is not None:
            self.positives = id_map(self.positives)
            self.negatives = id_map(self.negatives)


class StImageSelector(object):
    def __init__(self, imgs):
        self.imgs = imgs
        positive_checks = []
        negative_checks = []
        for i, img in enumerate(imgs):
            st.sidebar.image(img.resize((64, 64)))
            positive_checks.append(st.sidebar.checkbox(f'{i} +ve'))
            negative_checks.append(st.sidebar.checkbox(f'{i} -ve'))
        self.positives = np.arange(len(imgs))[positive_checks]
        self.negatives = np.arange(len(imgs))[negative_checks]


def get_plotly_scatter_highlight_selected(df, selected_idx=None):
    scatter = go.Figure()
    scatter.add_trace(go.Scatter3d(mode='markers',
                                   name='all',
                                   x=df[0], y=df[1], z=df[2],
                                   opacity=0.7,
                                   text=df.index.tolist(),
                                   marker={'size': 2}))
    if selected_idx is not None:
        scatter.add_trace(go.Scatter3d(mode='markers+text',
                                       name='selected',
                                       x=df[0].iloc[selected_idx], y=df[1].iloc[selected_idx],
                                       z=df[2].iloc[selected_idx],
                                       opacity=1.,
                                       text=selected_idx,
                                       textfont={'color': 'red'},
                                       textposition='top center',
                                       marker={'size': 5, 'color': 'red'}))
    scatter.update_layout(showlegend=False)
    return scatter


def st_plotly_weights_bar(weights, layer_names):
    df = pd.DataFrame(weights, columns=['weight'])
    df['layer'] = layer_names
    weights_bar = px.bar(df, x='layer', y='weight')
    weights_bar.update_layout()
    st.plotly_chart(weights_bar)


def weight_layers(grams, weights):
    weighted = []
    for gram, weight in zip(grams, weights):
        weighted.append(gram * weight)
    try:
        if isinstance(grams[0], torch.Tensor):
            return torch.cat(weighted, dim=-1)
    except:
        pass
    return np.concatenate(weighted, axis=-1)


class StQueryDisplay(object):
    def __init__(self, imgs, embedding, queries, query_idx, k=5, plot='plotly'):
        kd_tree = KDTree(embedding)
        knn_grid = KNNGridWithDistances(kd_tree, imgs, embedding, resize=256)
        if plot == 'plotly':
            self.grid = knn_grid.get_plotly(queries, query_idx=query_idx, k=k)
            st.plotly_chart(self.grid)
        elif plot == 'pyplot':
            self.grid = knn_grid.get_matplotlib(queries, query_idx=query_idx, k=k)
            st.pyplot(self.grid)


class IdMap(object):
    def __init__(self, src_file, dest_file):
        df = pd.read_csv(src_file, names=['name'])
        df = df.reset_index()
        df2 = pd.read_csv(dest_file, names=['name'])
        df2 = df2.reset_index()
        self.df = pd.merge(df, df2, on='name', how='outer')

    def __call__(self, idx):
        return self.df['index_y'].to_numpy(dtype=np.long)[idx]
