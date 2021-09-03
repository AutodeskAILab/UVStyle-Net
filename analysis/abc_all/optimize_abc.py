import re
import sys

import numpy as np
import plotly.graph_objs as go
import streamlit as st
import torch
import torchvision
from sklearn.utils import shuffle

sys.path.append('../../analysis')
sys.path.append('../../../UVStyle-Net')
from utils import top_k_neighbors, pad_grams, plot
from constrained_optimization import optimize
from util import Grams, OnTheFlyImages, get_pca_3_70

if __name__ == '__main__':
    device = torch.device('cuda:0')
    grams = Grams('../uvnet_data/abc_sub_mu_only')
    num_examples = len(grams.labels)

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='../cache/uvnet_abc_sub_mu_only',
                                 verbose=True)
    grams = list(pca_70.values())[:7]
    padded_grams = pad_grams(grams).to(device)

    images = OnTheFlyImages(data_root='../uvnet_data/abc_sub_mu_only',
                            img_root='../abc_pngs',
                            black_to_white=True)

    positives_text = st.sidebar.text_area(label='Positives (space separated ids):',
                                          value='24905 21560')
    positives_idx = list(map(int, re.split('\\s+', positives_text)))

    negatives_text = st.sidebar.text_area(label='Negatives (space separated ids):',
                                          value='27930')
    negatives_idx = list(map(int, re.split('\\s+', negatives_text))) if len(negatives_text) > 0 else []

    query_idx = positives_idx[0]

    st.sidebar.subheader('Positive Egs')
    for i in positives_idx:
        st.sidebar.image(images[i], width=200)

    st.sidebar.subheader('Negative Egs')
    for i in negatives_idx:
        st.sidebar.image(images[i], width=200)
    remaining = shuffle(list(set(np.arange(num_examples)).difference(positives_idx)))
    negatives_idx += remaining[:100]
    with st.spinner('optimizing...'):
        optimal_weights = optimize(positive_idx=positives_idx,
                                   negative_idx=negatives_idx,
                                   grams=grams,
                                   metric='cosine')

    combos = {
        'Uniform Layer Weights': torch.ones(7).to(device),
        'Optimized': torch.tensor(optimal_weights).to(device)
    }

    for name, weights in combos.items():
        st.subheader(name)
        print('compute neighbors...')
        neighbors, distances = top_k_neighbors(X=padded_grams,
                                               weights=weights,
                                               queries=[query_idx],
                                               k=6)

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
                   query_idx=[query_idx],
                   img_size=128,
                   k=6,
                   distances=distances.flatten())
        print('st plot...')
        st.pyplot(fig)

    st.subheader('Optimized Weights')
    fig = go.Figure(go.Bar(x=np.arange(len(optimal_weights)),
                           y=optimal_weights))
    st.plotly_chart(fig)
