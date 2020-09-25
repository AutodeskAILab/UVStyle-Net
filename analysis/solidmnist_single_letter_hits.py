import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

from constrained_optimization import optimize
from util import Grams, Images, get_pca_3_70, StImageSelector, get_plotly_scatter_highlight_selected, \
    st_plotly_weights_bar, weight_layers, StQueryDisplay


class Selector:
    def __init__(self, id_map=None):
        self.id_map = id_map
        self.pos_text = st.sidebar.text_area(label='Positives')
        self.neg_text = st.sidebar.text_area(label='Negatives')

    def _to_ids(self, text):
        idx = list(map(int, text.split(' '))) if len(text) > 0 else []
        if self.id_map is not None:
            idx = self.id_map(idx)
        return idx

    @property
    def positives(self):
        return self._to_ids(self.pos_text)

    @property
    def negatives(self):
        return self._to_ids(self.neg_text)


if __name__ == '__main__':
    grams = Grams(data_root='uvnet_data/solidmnist_single_letter')
    imgs = Images(data_root='uvnet_data/solidmnist_single_letter',
                  img_root='/Users/t_meltp/uvnet-img/png/test',
                  cache_file='cache/single_letter_pngs')

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='cache/uvnet_solidmnist_single_letter')

    selector = Selector()

    st.text(f'positives ({len(selector.positives)}): {selector.positives}')
    st.text(f'negatives ({len(selector.negatives)}): {selector.negatives}')

    button = st.button('Compute Queries')

    if button:
        for p in range(1, len(selector.positives) + 1):
            for n in range(len(selector.negatives) + 1):
                st.text(f'p: {p}, n: {n}')

                # perform weight optimization
                weights = optimize(positive_idx=selector.positives[:p],
                                   negative_idx=selector.negatives[:n],
                                   grams=list(pca_70.values()))

                embedding = weight_layers(pca_70.values(), weights)
                norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
                embedding = embedding / norm

                # QUERIES
                # get center of positive examples
                query_idx = selector.positives[:p]
                queries = embedding[query_idx, :].mean(0, keepdims=True)

                StQueryDisplay(imgs=imgs,
                               embedding=embedding,
                               queries=queries,
                               query_idx=query_idx,
                               k=10)
