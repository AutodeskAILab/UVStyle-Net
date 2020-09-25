import sys

from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from umap import UMAP
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

sys.path.append('/Users/t_meltp/OneDrive - autodesk/PycharmProjects/NURBSNet/analysis')
from util import Grams, weight_layers
from all_fonts_svms.all_fonts_svm import pca_reduce

if __name__ == '__main__':
    red = st.sidebar.text_area(label='Red:', value='44')
    blue = st.sidebar.text_area(label='Blue:', value='45')
    red_idx = list(map(int, red.split(' ')))
    blue_idx = list(map(int, blue.split(' ')))

    print('loading data...')
    grams = Grams('../uvnet_data/solidmnist_all')
    reduced = pca_reduce(grams, 18, '../cache/solidmnist_all')

    equal_weights = weight_layers(reduced, np.ones(len(grams)))

    labels = grams.labels
    # equal_weights, labels = shuffle(equal_weights, grams.labels)
    # equal_weights = equal_weights[:1000]
    # labels = labels[:1000]

    print('umap...')
    umap = UMAP(n_components=3, metric='cosine', verbose=1, n_epochs=500)
    equal_weights_emb = umap.fit_transform(equal_weights)

    print('scatter...')
    colors = np.ones(len(labels))
    for r_idx in red_idx:
        colors[labels == r_idx] = 0.
    for b_idx in blue_idx:
        colors[labels == b_idx] = .5

    scatter = go.Scatter3d(x=equal_weights_emb[:, 0],
                           y=equal_weights_emb[:, 1],
                           z=equal_weights_emb[:, 2],
                           hovertext=labels,
                           marker={
                               'colorscale': [
                                   [0., 'red'],
                                   [0.5, 'blue'],
                                   [1., 'rgba(0, 0, 0, 0.005)']
                               ],
                               'color': colors.tolist(),
                               'size': 5,
                           },
                           mode='markers')

    st.plotly_chart(go.Figure(scatter))
