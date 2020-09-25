import math
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier

from layer_stats_analysis import probe_score


def load_df(run, layer):
    pickle_file = os.path.join(run, layer, 'tensors.pd')
    if os.path.exists(pickle_file):
        df = pd.read_pickle(pickle_file)
    else:
        tensor_path = os.path.join(run, layer, 'tensors.tsv')
        df = pd.read_csv(tensor_path, header=None, sep='\t')
        df.to_pickle(pickle_file)
    return df


def get_image_grid(emb, metric, img_path, grid_size=9):
    all_img = Image.open(img_path)
    n_img = emb.shape[0]

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

    if metric == 'cosine':
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / norm
    kdtree = KDTree(emb)
    neighbours = kdtree.query(emb, k=5, return_distance=False)
    neighbours = neighbours[np.random.random_integers(0, n_img - 1, 70)]
    neighbours = neighbours.flatten()

    i = imgs[neighbours]

    grid = torchvision.utils.make_grid(i, nrow=5)
    fig = go.Figure(px.imshow(grid.permute((1, 2, 0)), height=8000, width=800))

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(n_img) * 52 + 27,
            ticktext=np.arange(n_img).tolist()
        )
    )
    return fig


def knn_score(emb, labels, metric):
    if metric == 'cosine':
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / norm
    clf = KNeighborsClassifier()
    score = cross_val_score(clf, emb, labels, scoring='accuracy', cv=5)
    return np.mean(score)


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

    st.header('All')
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

    scores_cache = 'layer_probing_scores_cache.pd'
    if os.path.exists(scores_cache):
        df[['sil_euc', 'sil_cos', 'knn_euc', 'knn_cos', 'lin_prob']] = \
            pd.read_pickle(scores_cache)
    else:
        df['sil_euc'] = pd.DataFrame.from_dict({
            i: silhouette_score(pca_70[run['rotation'], run['layer']], meta['font'], metric='euclidean')
            for i, run in df.to_dict('index').items()
        }, orient='index')

        df['sil_cos'] = pd.DataFrame.from_dict({
            i: silhouette_score(pca_70[run['rotation'], run['layer']], meta['font'], metric='cosine')
            for i, run in df.to_dict('index').items()
        }, orient='index')

        df['knn_euc'] = pd.DataFrame.from_dict({
            i: knn_score(pca_70[run['rotation'], run['layer']], meta['font'], metric='euclidean')
            for i, run in df.to_dict('index').items()
        }, orient='index')

        df['knn_cos'] = pd.DataFrame.from_dict({
            i: knn_score(pca_70[run['rotation'], run['layer']], meta['font'], metric='cosine')
            for i, run in df.to_dict('index').items()
        }, orient='index')

        df['lin_prob'] = pd.DataFrame.from_dict({
            i: probe_score(i, pca_70[run['rotation'], run['layer']], meta['font'])
            for i, run in df.to_dict('index').items()
        }, orient='index')

        df[['sil_euc', 'sil_cos', 'knn_euc', 'knn_cos', 'lin_prob']].to_pickle(scores_cache)

    st.dataframe(df.drop('path', axis=1))
    df = pd.concat([
        df[df['layer'].str.contains(layer_start)]
        for layer_start in ['feats', 'conv', 'fc', 'GIN']
    ])

    st.header('Raw Scores')
    gram_only = st.checkbox('Gram Only', value=True)
    score_metrics = ['sil_euc', 'sil_cos', 'knn_euc', 'knn_cos', 'lin_prob']
    fig = make_subplots(rows=len(score_metrics), cols=3,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        horizontal_spacing=0.01,
                        row_titles=score_metrics,
                        column_titles=['Original', 'Rotate x', 'Rotate z'])
    fig.update_layout(showlegend=False, height=800)
    for i, score_metric in enumerate(score_metrics):
        for j, (rotation, run) in enumerate(df.groupby('rotation')):
            if gram_only:
                run = run[run['layer'].str.contains('_in_gram')]
            bar = go.Bar(x=run['layer'], y=run[score_metric], name=rotation)
            fig.add_trace(bar, row=i + 1, col=j + 1)
    st.plotly_chart(fig)

    st.sidebar.header('Top k Query')
    rotation = st.sidebar.selectbox(label='Rotation', options=['original', 'rotate x', 'rotate z'])
    layer_sliders = {}
    for layer in filter(lambda l: l[-7:] == 'in_gram', df['layer'].unique()):
        layer_sliders[layer] = st.sidebar.slider(layer, min_value=0., max_value=1., step=.01)

    embs_to_concat = []
    for layer, value in layer_sliders.items():
        if value > 0.:
            embs_to_concat.append(pca_70[rotation, layer] * value)

    if len(embs_to_concat) > 0:
        st.header(f'3D Reduction ({len(embs_to_concat)} Layers)')
        combined = np.concatenate(embs_to_concat, axis=-1)
        pca3 = PCA(n_components=3)
        pca_3_reduced = pca3.fit_transform(combined)
        scatter = px.scatter_3d(pca_3_reduced, x=0, y=1, z=2, color=meta['font'])
        st.plotly_chart(scatter)

        sil_euc_combined = silhouette_score(combined, meta['font'], metric='euclidean')
        sil_cos_combined = silhouette_score(combined, meta['font'], metric='cosine')
        knn_euc_combined = knn_score(combined, meta['font'], metric='euclidean')
        knn_cos_combined = knn_score(combined, meta['font'], metric='cosine')
        probe_score_combined = probe_score(None, combined, meta['font'])
        st.write(f"Sihouette Score (Euclidean): {sil_euc_combined}")
        st.write(f"Sihouette Score (Cosine): {sil_cos_combined}")
        st.write(f"knn Score (Euclidean): {knn_euc_combined}")
        st.write(f"knn Score (Cosine): {knn_cos_combined}")
        st.write(f"Linear Probe Score: {probe_score_combined}")

        st.header(f'Top k Query for {len(embs_to_concat)} Layers')
        metric = st.radio('Metric', options=['cosine', 'euclidean'])
        img_grid = get_image_grid(combined, metric, img_path, grid_size=math.ceil(math.sqrt(len(meta))))
        st.plotly_chart(img_grid)
