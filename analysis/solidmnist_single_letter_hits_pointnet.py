import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

from constrained_optimization import optimize
from solidmnist_single_letter_hits import Selector
from util import Grams, Images, get_pca_3_70, StImageSelector, get_plotly_scatter_highlight_selected, \
    st_plotly_weights_bar, weight_layers, StQueryDisplay, IdMap, StImageSelectorTextBoxes

if __name__ == '__main__':
    data_root = 'pointnet2_data/solidmnist_single_letter'

    grams = Grams(data_root=data_root)
    imgs = Images(data_root=data_root,
                  img_root='/Users/t_meltp/uvnet-img/png/test',
                  cache_file='cache/single_letter_pngs_pointnet2')

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='cache/pointnet2_solidmnist_single_letter')

    id_map = IdMap(src_file='uvnet_data/solidmnist_single_letter/graph_files.txt',
                   dest_file=data_root + '/graph_files.txt')

    selector = Selector(id_map=id_map)

    st.text(f'positives ({len(selector.positives)}): {selector.positives}')
    st.text(f'negatives ({len(selector.negatives)}): {selector.negatives}')

    button = st.button('Compute Queries')

    if button:
        for p in range(1, len(selector.positives)+1):
            for n in range(len(selector.positives)+1):
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
                query_idx = selector.positives
                queries = embedding[query_idx, :].mean(0, keepdims=True)

                StQueryDisplay(imgs=imgs,
                               embedding=embedding,
                               queries=queries,
                               query_idx=query_idx,
                               k=10)
