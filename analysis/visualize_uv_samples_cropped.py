import streamlit as st
import dgl
import sys

sys.path.append('../')
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from solid_mnist import SolidMNIST, RandomCrop, identity_transform
import numpy as np


def get_sample(dset, id):
    graph = list(dset[id][0])[0]  # type: dgl.DGLGraph
    num_nodes = graph.number_of_nodes()
    return graph_to_xyz_mask(graph), num_nodes


if __name__ == '__main__':
    original = SolidMNIST(root_dir='../dataset/bin',
                          split='test')

    cropped = SolidMNIST(root_dir='../dataset/bin',
                         split='test',
                         transform=identity_transform,
                         crop_func=RandomCrop())

    id = 100

    # Original
    graph = original[id][0]
    xyz, mask = graph_to_xyz_mask(graph)
    plot = uv_samples_plot(xyz=xyz, mask=mask, node_colors=np.arange(graph.number_of_nodes()))
    st.plotly_chart(plot)

    # Cropped
    for i in range(3):
        (xyz, mask), num_nodes = get_sample(cropped, id)
        plot = uv_samples_plot(xyz=xyz, mask=mask, node_colors=np.arange(num_nodes))
        st.plotly_chart(plot)
