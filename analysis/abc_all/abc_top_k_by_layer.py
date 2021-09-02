import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances

sys.path.append('../../analysis')
from util import Grams, get_pca_3_70, OnTheFlyImages


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


def pad_grams(X):
    grams_0 = torch.zeros(len(X[0]), 70)
    grams_0[:, :X[0].shape[-1]] = torch.tensor(X[0].copy())
    grams_padded = torch.stack([grams_0] + [torch.tensor(gram.copy()) for gram in X[1:]])
    X = grams_padded.permute(1, 0, 2)  # shape: N x 7 x 70
    return X


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


if __name__ == '__main__':
    device = torch.device('cuda:0')
    data_root = '../uvnet_data/abc_sub_mu_only'
    # data_root = '../psnet_data/abc_all'
    grams = Grams(data_root=data_root)
    num = len(grams.graph_files)
    images = OnTheFlyImages(data_root=data_root,
                            img_root='../abc_pngs',
                            black_to_white=True)

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams.graph_files))
    }

    pca_3, pca_70 = get_pca_3_70(grams,
                                 cache_file='../cache/uvnet_abc_sub_mu_only',
                                 # cache_file='../cache/psnet_abc_all',
                                 verbose=True)
    del grams

    text_idx_names = st.sidebar.text_area(label='Enter file names (one per line)',
                                          value='\'16Regs - 2016RegulationBox\n'
                                                '1_10th_scale_on_road_car Materials v5 v1 v0 parts - Part 121\n'
                                                '2 - Part 1-8g5pl30u\n'
                                                '3 Fillet - Part 1\n'
                                                'Lego - Part 1')
    query_idx = list(map(lambda n: name_idx[n], text_idx_names.split('\n')))

    # defaults = [1., 1., 1., 1., 0., 0., 0.]
    # weights = [st.sidebar.slider(label=str(i),
    #                              min_value=0.,
    #                              max_value=1.,
    #                              step=0.01,
    #                              value=defaults[i])
    #            for i in range(len(defaults))]
    # weight_combos = np.array([weights])

    # weight_combos = torch.eye(7).to(device)
    weight_combos = torch.cat([torch.tensor([[1., 1., 0., 0., 0., 0., 0.]], device=device),
                               torch.tensor([[1., 1., 1., 0., 0., 0., 0.]], device=device),
                               torch.tensor([[1., 1., 1., 1., 0., 0., 0.]], device=device),
                               torch.tensor([[1., 1., 1., 1., 1., 0., 0.]], device=device),
                               torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=device)])
    padded_grams = pad_grams(list(pca_70.values())).to(device)

    for layer, weights in enumerate(weight_combos):
        print('compute neighbors...')
        neighbors, distances = top_k_neighbors(X=padded_grams,
                                               weights=weights,
                                               queries=query_idx,
                                               k=6,
                                               metric='cosine')

        st.subheader('weights')
        st.write(weights)
        st.subheader(f'ids')
        st.write(neighbors)

        print('map to tensors')
        imgs = images[neighbors.flatten()]
        t = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        img_tensors = list(map(t, imgs))
        print('make grid...')
        grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))
        print('make figure...')
        fig = plot(grid_array=grid,
                   query_idx=['A', 'B', 'C', 'D', 'E'],
                   img_size=128,
                   k=6,
                   distances=distances.flatten(),
                   font_size=36)
        print('st plot...')
        st.pyplot(fig)
        fig.savefig(f'uvnet_{layer}.pdf')
