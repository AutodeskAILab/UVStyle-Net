import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

from constrained_optimization import optimize
from util import Grams, Images, get_pca_3_70, StImageSelector, get_plotly_scatter_highlight_selected, \
    st_plotly_weights_bar, weight_layers, StQueryDisplay

if __name__ == '__main__':
    grams = Grams(data_root='uvnet_data/abc')
    imgs = Images(data_root='uvnet_data/abc',
                  img_root='/Users/t_meltp/abc/pngs/test',
                  cache_file='cache/abc_pngs')

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='cache/uvnet_abc_raw_grams_pcas')

    text_pos_idx = st.sidebar.text_area(label='Enter space separated +ve ids',
                                        value='6355 5414')
    text_neg_idx = st.sidebar.text_area(label='Enter space separated +ve ids',
                                        value='7955')
    pos_idx = list(map(int, text_pos_idx.split(' ')))
    neg_idx = list(map(int, text_neg_idx.split(' '))) if text_neg_idx else []

    if len(pos_idx) > 0:
        # perform weight optimization
        weights = optimize(positive_idx=pos_idx,
                           negative_idx=neg_idx,
                           grams=list(pca_70.values()))
    else:
        weights = np.ones(len(grams)) / len(grams)

    embedding = weight_layers(pca_70.values(), weights)

    st.subheader('UVNet Layer Weights')
    st_plotly_weights_bar(weights, pca_70.keys())

    pca = PCA(n_components=70)
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    embedding = embedding / norm
    embedding = pca.fit_transform(embedding)

    # QUERIES
    if len(pos_idx) > 0:
        # get center of positive examples
        query_idx = pos_idx
        queries = embedding[query_idx, :].mean(0, keepdims=True)

        pca = PCA(n_components=3)
        df = pd.DataFrame(pca.fit_transform(embedding))
        st.plotly_chart(get_plotly_scatter_highlight_selected(df, query_idx))

        StQueryDisplay(imgs=imgs,
                       embedding=embedding,
                       queries=queries,
                       query_idx=query_idx,
                       k=10)
