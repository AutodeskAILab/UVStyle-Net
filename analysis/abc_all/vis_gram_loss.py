import sys

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objs as go

sys.path.append('../../analysis')
from util import Grams, OnTheFlyImages


def gram_loss(idx_a, idx_b, grams):
    losses = []
    for gram in grams:
        gram_a = gram[idx_a]
        gram_b = gram[idx_b]
        loss = cosine_distances(gram_a[None, :], gram_b[None, :])
        losses.append(loss)
    return np.concatenate(losses).flatten()


if __name__ == '__main__':
    data_root = '../uvnet_data/abc_all_fnorm_only'
    grams = Grams(data_root)
    imgs = OnTheFlyImages(data_root=data_root,
                          img_root='../abc_pngs')

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams.graph_files))
    }

    base_text = st.sidebar.text_input(label='Name of solid to compare:',
                                      value='3 Fillet - Part 1')
    compare_text = st.sidebar.text_area(label='Solids to compare against (1 per line):',
                                        value='4 Chamfer - Part 1\n'
                                              'Part Studio 1 - Corner 8\n'
                                              'Part Studio 14 (6mm) - Part 1\n'
                                              '11 - Part 1\n'
                                              'Lego - Part 1')

    base_idx = name_idx[base_text]
    compare_idx = list(map(lambda n: name_idx[n], compare_text.split('\n')))

    st.sidebar.image(imgs[base_idx], output_format='PNG', width=300)

    for idx in [base_idx] + compare_idx:
        st.image(imgs[idx], output_format='PNG', width=300)
        losses = gram_loss(base_idx, idx, grams)
        bar = go.Bar(x=np.arange(len(losses)),
                     y=losses)
        fig = go.Figure(bar)
        fig.update_layout({
            'yaxis': {
                'range': [0, .52]
            }
        })
        st.plotly_chart(fig)
