import sys
from random import shuffle

import streamlit as st

sys.path.append('../')
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from solid_mnist import SolidMNISTSubset
import numpy as np
import pandas as pd


def plot(file):
    idx = df[df['file'] == file].index.values[0]
    graph = dset[idx][0]
    xyz, mask = graph_to_xyz_mask(graph)
    colors = np.arange(graph.number_of_nodes())
    shuffle(colors)
    plot = uv_samples_plot(xyz=xyz, mask=mask, node_colors=colors, marker_size=4)
    plot.update_layout({'scene': {
        'xaxis': {
            'zeroline': False,
            'showline': False,
            'showgrid': False,
            'showbackground': False,
            'visible': False
        },
        'yaxis': {
            'zeroline': False,
            'showline': False,
            'showgrid': False,
            'showbackground': False,
            'visible': False
        },
        'zaxis': {
            'zeroline': False,
            'showline': False,
            'showgrid': False,
            'showbackground': False,
            'visible': False
        }
    }})
    st.plotly_chart(plot)


if __name__ == '__main__':
    dset = SolidMNISTSubset(root_dir='../dataset/bin',
                            split='test')

    files = map(lambda n: n.stem, dset.graph_files)
    df = pd.DataFrame({'file': files})

    plot('l_Seaweed Script_upper')
    plot('z_Abhaya Libre_upper')

