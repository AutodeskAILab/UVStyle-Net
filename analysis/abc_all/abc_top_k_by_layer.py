import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torchvision
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances
from tqdm import tqdm

sys.path.append('../../analysis')
from util import Grams, get_pca_3_70, OnTheFlyImages


def plot(grid_array, queries, img_size, k, with_distance=True):
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.imshow(grid_array)

    ax.set_xticks(np.arange(k) * (img_size + 2) + (img_size / 2))
    ax.set_xticklabels(['Q'] + list(map(str, np.arange(1, k))), Fontsize=24)

    if query_idx is not None and len(query_idx) > 0:
        ax.set_yticks(np.arange(len(queries)) * (img_size + 2) + (img_size / 2))
        ax.set_yticklabels(query_idx if query_idx is not None else None, Fontsize=24)
    else:
        ax.set_yticks([])

    if with_distance:
        x = np.arange(k)
        y = np.arange(len(all_distances))
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        text = list(map(lambda d: f'{d:.2f}', all_distances.tolist()))

        space = img_size
        for t, pos in zip(text, xy):
            x, y = pos
            ax.annotate(t, ((x * space + .6 * space), (y * space + .9 * space)), color='red', Fontsize=20)

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


if __name__ == '__main__':
    data_root = '../uvnet_data/abc_sub_mu_only'
    grams = Grams(data_root=data_root)
    num = len(grams.graph_files)
    images = OnTheFlyImages(data_root=data_root,
                            img_root='../abc_pngs',
                            black_to_white=True)

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams.graph_files))
    }

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='../cache/uvnet_abc_sub_mu_only',
                                 verbose=True)
    del grams

    text_idx_names = st.sidebar.text_area(label='Enter file names (one per line)',
                                          value='Camber 1.5 2.5 3.5 - Part 1\n'
                                                '\'16Regs - 2016RegulationBox\n'
                                                '+ - Part 1-z17r786il4gl17a\n'
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

    weight_combos = np.array([
        [1., 1., 1., 1., 0., 0., 0.],
        # [1., 1., 1., 0., 0., 0., 0.],
        # [1., 1., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 1.]
    ])
    for layer, weights in enumerate(weight_combos):
        print('weight layers...')
        results = []
        all_distances = []
        for query in query_idx:
            st.text(query)
            distances = np.zeros(num)
            inputs = tqdm(range(num))
            x = Parallel(-1)(delayed(gram_loss)(list(pca_70.values()), query, other, weights, metric='euclidean') for other in inputs)
            for idx, distance in x:
                distances[idx] = distance
            results.append(np.argsort(distances)[:6])
            all_distances.append(distances[np.argsort(distances)[:6]])
        all_results = np.concatenate(results, axis=-1)
        all_distances = np.concatenate(all_distances, axis=-1)

        print('map to tensors')
        imgs = images[all_results]
        t = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        img_tensors = list(map(t, imgs))
        print('make grid...')
        grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))
        fig = plot(grid, query_idx, 128, 6, True)
        print('st plot...')
        st.pyplot(fig)
