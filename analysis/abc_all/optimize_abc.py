import sys

import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.decomposition import PCA

sys.path.append('../../analysis')
from constrained_optimization import optimize
from util import Grams, OnTheFlyImages, StQueryDisplay, weight_layers, get_pca_3_70


if __name__ == '__main__':
    grams = Grams('../uvnet_data/abc_sub_mu_only')

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='../cache/uvnet_abc_sub_mu_only',
                                 verbose=True)
    grams = list(pca_70.values())

    imgs = OnTheFlyImages(data_root='../uvnet_data/abc_sub_mu_only',
                          img_root='../abc_pngs',
                          black_to_white=True)

    positives_text = st.sidebar.text_area(label='Positives (space separated ids):',
                                          value='25460 22040')
    positives_idx = list(map(int, positives_text.split(' ')))

    negatives_text = st.sidebar.text_area(label='Negatives (space separated ids):',
                                          value='28541')
    negatives_idx = list(map(int, negatives_text.split(' '))) if len(negatives_text) > 0 else []

    query_text = st.sidebar.text_input(label='query_idx',
                                       value='25460')
    query_idx = int(query_text)

    st.sidebar.subheader('Positive Egs')
    for i in positives_idx:
        st.sidebar.image(imgs[i], width=200)

    st.sidebar.subheader('Negative Egs')
    for i in negatives_idx:
        st.sidebar.image(imgs[i], width=200)

    with st.spinner('optimizing...'):
        weights = optimize(positive_idx=positives_idx,
                           negative_idx=negatives_idx,
                           grams=grams,
                           metric='cosine')

    st.subheader('Optimized Weights')
    fig = go.Figure(go.Bar(x=np.arange(len(weights)),
                           y=weights))
    st.plotly_chart(fig)

    print('equal...')
    equal = weight_layers(grams, np.ones_like(weights))
    StQueryDisplay(imgs, equal, equal[[query_idx]], [], 11, plot='pyplot')

    st.subheader('Top-10 Queries')
    combined = weight_layers(grams, weights)

    del grams
    reduced = combined

    print('queries...')
    norm = np.linalg.norm(reduced, axis=-1, keepdims=True)
    x = reduced / norm
    StQueryDisplay(imgs=imgs,
                   embedding=x,
                   queries=x[[query_idx]],
                   query_idx=[],
                   k=11,
                   plot='pyplot')
