import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

from constrained_optimization import optimize
from util import Grams, Images, get_pca_3_70, StImageSelector, get_plotly_scatter_highlight_selected, \
    st_plotly_weights_bar, weight_layers, StQueryDisplay

if __name__ == '__main__':
    grams = Grams(data_root='pointnet2_data/solidmnist_single_letter')
    imgs = Images(data_root='pointnet2_data/solidmnist_single_letter',
                  img_root='/Users/t_meltp/uvnet-img/png/test',
                  cache_file='cache/single_letter_pngs_pointnet2')

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='cache/pointnet2_solidmnist_single_letter')

    text_idx = st.sidebar.text_area(label='Enter space separated query ids or \'mean\' for center of positive eg.s',
                                    value='7 2 11 89 24')
    selector = StImageSelector(imgs)

    # selector.positives = [1, 2]
    if len(selector.positives) > 0:
        # perform weight optimization
        weights = optimize(positive_idx=selector.positives,
                           negative_idx=selector.negatives,
                           grams=list(pca_70.values()))
    else:
        weights = np.ones(len(grams)) / len(grams)

    embedding = weight_layers(pca_70.values(), weights)

    st.subheader('Pointnet++ Layer Weights')
    st_plotly_weights_bar(weights, pca_70.keys())

    pca = PCA(n_components=70)
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    embedding = embedding / norm
    embedding = pca.fit_transform(embedding)

    # QUERIES
    if text_idx == 'mean':
        # get center of positive examples
        query_idx = selector.positives
        queries = embedding[query_idx, :].mean(0, keepdims=True)
    else:
        query_idx = list(map(int, text_idx.split(' ')))
        queries = embedding[query_idx]

    pca = PCA(n_components=3)
    df = pd.DataFrame(pca.fit_transform(embedding))
    st.plotly_chart(get_plotly_scatter_highlight_selected(df, query_idx))

    StQueryDisplay(imgs=imgs,
                   embedding=embedding,
                   queries=queries,
                   query_idx=query_idx,
                   k=7)
