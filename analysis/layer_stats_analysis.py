import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree


@st.cache
def get_image_grid(stat, metric):
    all_img = Image.open(os.path.join(run_dir, stat, 'sprite.png'))
    n_img = len(meta)

    def r_c(k):
        r = k // 9
        c = k % 9
        return r, c

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
    emb = pca_70[stat]
    if metric == 'cosine':
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / norm
    kdtree = KDTree(emb)
    neighbours = kdtree.query(pca_70[stat], k=5, return_distance=False)
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


def load_embeddings(stat, run_dir):
    print(f'loading {stat}...')
    binary_cache = os.path.join(run_dir, stat, 'tenors.pd')
    if os.path.exists(binary_cache):
        emb = pd.read_pickle(binary_cache)
    else:
        emb = pd.read_csv(os.path.join(run_dir, stat, 'tensors.tsv'), sep='\t', header=None)
        emb.to_pickle(binary_cache)
    return emb


def pca(raw_emb, n_components):
    pca_ = PCA(n_components=n_components)
    return pca_.fit_transform(raw_emb)


@st.cache
def probe_score(stat, reduced, labels, err=False, balanced=True):
    print(f'Probing {stat}...')
    model = LogisticRegressionCV(cv=5,
                                 class_weight='balanced' if balanced else None,
                                 scoring='accuracy',
                                 max_iter=1000)
    model.fit(reduced, labels)
    scores = []
    if err:
        errs = []
        for k, v in model.scores_.items():
            c_scores = v.mean(axis=0)
            best_c = c_scores.argmax()
            scores.append(c_scores.max())
            errs.append(v[:, best_c].std())
        return np.mean(scores), np.mean(errs)
    else:
        for k, v in model.scores_.items():
            c_scores = v.mean(axis=0)
            scores.append(c_scores.max())
        return np.mean(scores)


@st.cache()
def get_pca():
    pca_3 = {
        stat: pca(raw_emb, 3) for stat, raw_emb in raw_embeddings.items()
    }
    pca_70 = {
        stat: pca(raw_emb, 70) if raw_emb.shape[-1] > 70 else raw_emb
        for stat, raw_emb in raw_embeddings.items()
    }
    return pca_3, pca_70


if __name__ == '__main__':
    run_dir = '../runs/Jul01_12-47-17_C02ZX3Q7MD6R/00000'
    stats = os.listdir(run_dir)

    with st.spinner('Loading data...'):
        meta = pd.read_csv(os.path.join(run_dir, stats[0], 'metadata.tsv'), sep='\t')
        raw_embeddings = {
            stat: load_embeddings(stat, run_dir) for stat in stats
        }

    with st.spinner('Compute PCA...'):
        pca_3, pca_70 = get_pca()

    with st.spinner('Compute Silhouette Scores...'):
        euclidean_silhouette_scores = {
            stat: silhouette_score(reduced, meta['font']) for stat, reduced in pca_70.items()
        }
        cosine_silhouette_scores = {
            stat: silhouette_score(reduced, meta['font'], metric='cosine') for stat, reduced in pca_70.items()
        }
        df = pd.DataFrame.from_dict(euclidean_silhouette_scores,
                                    orient='index', columns=['Silhouette Score (Euclidean)'])
        df['Silhouette Score (Cosine)'] = pd.DataFrame.from_dict(cosine_silhouette_scores,
                                                                 orient='index')

    with st.spinner('Compute Silhouette Scores...'):
        probe_scores = {
            stat: probe_score(stat, reduced, meta['font']) for stat, reduced in pca_70.items()
        }
        df['Probe Score'] = pd.DataFrame.from_dict(probe_scores, orient='index')

    print(df.sort_values('Silhouette Score (Euclidean)', ascending=False).to_markdown())

    st.header("Layer Scores")
    st.dataframe(df)

    option = st.sidebar.selectbox('run', options=sorted(stats))

    # show 3d PCA plot
    st.subheader(option)
    plot = px.scatter_3d(pca_3[option], x=0, y=1, z=2, color=meta['font'], text=meta.index)
    st.plotly_chart(plot)

    # show queries
    metric = st.radio('distance metric', options=['euclidean', 'cosine'])
    with st.spinner('Computing Queries...'):
        img_grid = get_image_grid(option, metric)
        st.plotly_chart(img_grid)
