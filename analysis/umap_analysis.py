import math
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torchvision
import umap

from layer_probing import get_image_grid
from optimization import get_img

if __name__ == '__main__':
    run_dir = '../runs/Jul03_11-24-39_C02ZX3Q7MD6R/00000'
    img_path = '../runs/Jul03_11-24-39_C02ZX3Q7MD6R/00000/conv1_in_cov/sprite.png'
    stats = os.listdir(run_dir)

    with st.spinner('Loading data...'):
        meta = pd.read_csv(os.path.join(run_dir, stats[0], 'metadata.tsv'), sep='\t')
        # raw_embeddings = {
        #     stat: load_embeddings(stat, run_dir) for stat in stats
        # }

    cache_file = 'pca_cache_4_fonts.pickle'
    with open(cache_file, 'rb') as file:
        pca = pickle.load(file)

    pca_3, pca_70 = pca
    pca_30 = {name: v for (rotation, name), v in pca_3.items() if rotation == 'original' and name[-4:] == 'gram'}
    pca_70 = {name: v for (rotation, name), v in pca_70.items() if rotation == 'original' and name[-4:] == 'gram'}

    imgs = get_img(len(pca_70['feats_in_gram']), img_path)

    positive_checks = []
    negative_checks = []
    for i, img in enumerate(imgs):
        st.sidebar.image(torchvision.transforms.ToPILImage()(img))
        positive_checks.append(st.sidebar.checkbox(f'{i} +ve ({meta["font"][i]})'))
        negative_checks.append(st.sidebar.checkbox(f'{i} -ve ({meta["font"][i]})'))

    labels = - np.ones(len(imgs), dtype=np.long)
    labels[positive_checks] = 0
    labels[negative_checks] = 1

    data = list(pca_70.values())
    data = np.concatenate(data, axis=-1)

    st.header('UMAP Semi-Supervised')
    st.write(f'{sum(positive_checks)} positive examples selected')
    st.write(f'{sum(negative_checks)} negative examples selected')

    fitter = umap.UMAP(n_components=3).fit(data, y=labels)
    embedding_3 = fitter.embedding_
    scatter = px.scatter_3d(x=embedding_3[:, 0],
                            y=embedding_3[:, 1],
                            z=embedding_3[:, 2],
                            color=meta['font'],
                            )
    st.write(scatter)

    fitter = umap.UMAP(n_components=50).fit(data, y=labels)
    embedding_50 = fitter.embedding_

    st.subheader('k-NN Query')
    img_grid = get_image_grid(embedding_50, 'cosine', img_path, grid_size=math.ceil(math.sqrt(len(meta))))
    st.plotly_chart(img_grid)