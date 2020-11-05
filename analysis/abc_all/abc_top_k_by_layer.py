import sys

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

sys.path.append('../../analysis')
from util import Grams, get_pca_3_70, StQueryDisplay, weight_layers, OnTheFlyImages

if __name__ == '__main__':
    data_root = '../uvnet_data/abc_sub_mu_only'
    grams = Grams(data_root=data_root)
    imgs = OnTheFlyImages(data_root=data_root,
                          img_root='../abc_pngs',
                          black_to_white=True)

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams.graph_files))
    }

    pca_3, pca_70 = get_pca_3_70(grams, cache_file='../cache/uvnet_abc_sub_mu_only',
                                 verbose=True)

    combined = np.concatenate(list(pca_70.values()), axis=-1)

    text_idx_names = st.sidebar.text_area(label='Enter file names (one per line)',
                                          value='Camber 1.5 2.5 3.5 - Part 1\n'
                                                '\'16Regs - 2016RegulationBox\n'
                                                '+ - Part 1-z17r786il4gl17a\n'
                                                '1_10th_scale_on_road_car Materials v5 v1 v0 parts - Part 121\n'
                                                '2 - Part 1-8g5pl30u\n'
                                                '3 Fillet - Part 1\n'
                                                'Lego - Part 1')
    query_idx = list(map(lambda n: name_idx[n], text_idx_names.split('\n')))

    defaults = [1., 1., 1., 1., 0., 0., 0.]
    weights = [st.sidebar.slider(label=str(i),
                                 min_value=0.,
                                 max_value=1.,
                                 step=0.01,
                                 value=defaults[i])
               for i in range(len(grams))]

    weight_combos = np.array([weights])

    for layer, weights in enumerate(weight_combos):
        print('weight layers...')
        combined = weight_layers(pca_70.values(), weights)

        print('pca on weighted layers...')
        pca = PCA(n_components=70)
        combined = pca.fit_transform(combined)
        norm = np.linalg.norm(combined, axis=-1, keepdims=True)
        combined = combined / norm

        print('top-k queries...')
        st.write(f'Layer {layer}')
        query_display = StQueryDisplay(imgs=imgs,
                              embedding=combined,
                              queries=combined[query_idx],
                              query_idx=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                              k=11,
                              plot='pyplot')
        query_display.grid.savefig(f'abc_queries_{layer}.pdf')