import math
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
from PIL import Image
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from layer_probing import load_df, get_image_grid
from layer_stats_analysis import probe_score


def get_img(n_img, img_path, grid_size=9):
    all_img = Image.open(img_path)

    def r_c(k):
        r = k // grid_size
        c = k % grid_size
        return r, c

    img_cache = 'layer_probe_img_cache.pt'
    if os.path.exists(img_cache):
        imgs = torch.load(img_cache)
    else:
        imgs = []
        for i in range(n_img):
            r, c = r_c(i)
            # image size 480 x 480
            s = 480
            # box = (left, top, right, bottom)
            box = (s * c, s * r, s * c + s, s * r + s)
            img = all_img.crop(box).resize([50, 50])
            imgs.append(torchvision.transforms.ToTensor()(img).numpy())

        imgs = torch.tensor(imgs)
        torch.save(imgs, img_cache)
    return imgs


def objective(w):
    positive_loss = 0
    negative_loss = 0
    num_pos = sum(positive_checks)
    num_neg = sum(negative_checks)
    for l in range(len(gram_layers)):
        for i, i_v in enumerate(positive_checks):
            for j, j_v in enumerate(positive_checks):
                if i_v and j_v and i < j:
                    positive_loss += (w[l] * np.linalg.norm(grams[l][i] - grams[l][j])) \
                                     / (4 * (grams[l].shape[-1] ** 2))
        if sum(negative_checks) > 0:
            for i, i_v in enumerate(positive_checks):
                for j, j_v in enumerate(negative_checks):
                    if i_v and j_v:
                        negative_loss += (w[l] * np.linalg.norm(grams[l][i] - grams[l][j])) \
                                         / (1 * (grams[l].shape[-1] ** 2))

    positive_loss *= 2 / (num_pos * (num_pos + 1))
    if sum(negative_checks) > 0:
        negative_loss *= 1 / (num_pos * num_neg)
    return positive_loss - negative_loss


def constraint(w):
    return w.sum() - 1


if __name__ == '__main__':

    root_path = '../'
    paths = {
        'original': 'runs/Jul03_11-24-39_C02ZX3Q7MD6R',
        'rotate x': 'runs/Jul03_11-32-21_C02ZX3Q7MD6R',
        'rotate z': 'runs/Jul03_11-37-44_C02ZX3Q7MD6R',
    }
    df = pd.DataFrame.from_dict(paths, orient='index').reset_index()
    df.columns = ['rotation', 'path']
    df['path'] = df['path'].apply(lambda p: os.path.join(root_path, p, '00000'))

    meta_data_path = os.path.join(df.iloc[0]['path'], 'conv1_in_cov', 'metadata.tsv')
    img_path = os.path.join(df.iloc[0]['path'], 'conv1_in_cov', 'sprite.png')

    layers = sorted(os.listdir(df.iloc[0]['path']))
    ldf = pd.DataFrame(layers, columns=['layer'])
    ldf['key'] = 1
    df['key'] = 1
    df = df.merge(ldf, how='outer', on='key')
    df.drop('key', axis=1, inplace=True)

    meta = pd.read_csv(meta_data_path, sep='\t')

    raw_cache = 'layer_probe_raw_cache.pickle'
    if os.path.exists(raw_cache):
        with open(raw_cache, 'rb') as file:
            raw = pickle.load(file)
    else:
        all_bar = st.progress(0)
        raw = {}
        for i, run in df.to_dict('index').items():
            raw[run['rotation'], run['layer']] = load_df(run['path'], run['layer'])
            all_bar.progress((i + 1) / len(df))
        with open(raw_cache, 'wb') as file:
            pickle.dump(raw, file)
        all_bar.empty()

    pca_cache = f'pca_cache_4_fonts.pickle'
    if os.path.exists(pca_cache):
        with open(pca_cache, 'rb') as file:
            pca_3, pca_70 = pickle.load(file)
    else:
        pca3 = PCA(n_components=3)
        pca70 = PCA(n_components=70)
        pca_3 = {}
        pca_70 = {}
        pca_bar = st.progress(0)
        for i, (rotation_layer, layer) in enumerate(raw.items()):
            pca_3[rotation_layer] = pca3.fit_transform(layer)
            pca_70[rotation_layer] = pca70.fit_transform(layer) if layer.shape[-1] > 70 else layer
            print((i + 1) / len(raw))
            pca_bar.progress((i + 1) / len(raw))
        with open(pca_cache, 'wb') as file:
            pickle.dump((pca_3, pca_70), file)
        pca_bar.empty()

    imgs = get_img(len(pca_70['original', 'feats_in_cov']), img_path)

    gram_layers = ['feats_in_gram', 'conv1_in_gram', 'conv2_in_gram', 'conv3_in_gram',
                   'fc_in_gram', 'GIN_1_in_gram', 'GIN_2_in_gram']
    grams = []
    for gram_layer in gram_layers:
        grams.append(pca_70['original', gram_layer])

    grams[0] = grams[0].to_numpy()

    positive_checks = []
    negative_checks = []
    for i, img in enumerate(imgs):
        st.sidebar.image(torchvision.transforms.ToPILImage()(img))
        positive_checks.append(st.sidebar.checkbox(f'{i} +ve'))
        negative_checks.append(st.sidebar.checkbox(f'{i} -ve'))

    st.write(f'{sum(positive_checks)} positive examples selected')
    st.write(f'{sum(negative_checks)} negative examples selected')
    if sum(positive_checks) > 0:
        # contrastive_loss(grams, np.array([5, 7, 10]), np.array([9, 11]))
        bounds = [[0., 1.]] * len(gram_layers)
        sol = minimize(objective, np.array([1 / len(gram_layers)] * len(gram_layers)),
                       bounds=bounds,
                       constraints={'type': 'eq', 'fun': constraint})
        st.write(sol.message)
        st.write(sol.x)

        st.subheader('Linear Probe')
        embs_to_concat = []
        for layer, value in enumerate(sol.x):
            embs_to_concat.append(pca_70['original', gram_layers[layer]] * value)
        combined = np.concatenate(embs_to_concat, axis=-1)
        probe_score_combined = probe_score(None, combined, meta['font'])
        st.write(f'Score: {probe_score_combined}')
        print(sol.x)

        st.subheader('k-NN Query')
        img_grid = get_image_grid(combined, 'euclidean', img_path, grid_size=math.ceil(math.sqrt(len(meta))))
        st.plotly_chart(img_grid)
