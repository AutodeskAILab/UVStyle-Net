import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import trimesh
from sklearn.preprocessing import StandardScaler

from datasets.solid_mnist import SolidMNISTSubset


def face_adjacency_plot(graph, node_colors=None):
    x = graph.ndata['x']
    df = pd.DataFrame(x[:, :, :, :3].flatten(end_dim=-2).numpy(), columns=['x', 'y', 'z'])
    df[['normal_x', 'normal_y', 'normal_z']] = pd.DataFrame(x[:, :, :, 3:6].flatten(end_dim=-2).numpy())
    df['mask'] = x[:, :, :, 6].reshape([-1, 1]).numpy()

    df['node'] = np.array([[i] * 100 for i in range(graph.number_of_nodes())]).flatten()

    masked = df[df['mask'] == 1.]

    # face adjacency graph
    face_adj = masked.groupby('node').mean()
    if node_colors is None:
        node_colors = face_adj.index
    src, dst, _ = zip(*list(graph.to_networkx().edges))
    src = face_adj.loc[list(src)]
    dst = face_adj.loc[list(dst)]
    face_adj_fig = go.Figure(data=go.Scatter3d(x=face_adj['x'],
                                               y=face_adj['y'],
                                               z=face_adj['z'],
                                               marker=dict(
                                                   color=node_colors
                                               ),
                                               mode='markers'))
    for i in range(len(src)):
        face_adj_fig.add_scatter3d(
            # Line reference to the axes
            x=[src.iloc[i]['x'], dst.iloc[i]['x']],
            y=[src.iloc[i]['y'], dst.iloc[i]['y']],
            z=[src.iloc[i]['z'], dst.iloc[i]['z']],
            mode='lines',
            line=dict(
                color="Black",
                width=1,
            )
        )
    face_adj_fig.update_layout({
        'showlegend': False
    })
    return face_adj_fig


def graph_to_xyz_mask(graph):
    x = graph.ndata['x']
    xyz = x[:, :, :, :3].reshape([-1, 3])
    mask = x[:, :, :, 6].reshape([-1])
    return xyz, mask


def uv_samples_plot(xyz, mask,
                    xyz_grads=None,
                    scale_xyz_grads=0.05,
                    marker_size=3,
                    mesh: trimesh.Trimesh = None,
                    mesh_alpha=1.):

    if mask is not None:
        xyz = xyz[mask == 1]

    scatter = go.Scatter3d(x=xyz[:, 0],
                           y=xyz[:, 1],
                           z=xyz[:, 2],
                           marker=dict(
                               size=marker_size,
                               color='rgb(0, 0, 255)'
                           ),

                           mode='markers')
    fig = go.Figure(data=scatter)

    if mesh is not None:
        mesh.vertices -= mesh.bounding_box.bounds[0]
        mesh_plot = go.Mesh3d(x=mesh.vertices[:, 0],
                              y=mesh.vertices[:, 1],
                              z=mesh.vertices[:, 2],
                              i=mesh.faces[:, 0],
                              j=mesh.faces[:, 1],
                              k=mesh.faces[:, 2],
                              color='gray',
                              opacity=mesh_alpha)
        fig.add_trace(mesh_plot)

    if xyz_grads is not None:
        if mask is not None:
            xyz_grads = xyz_grads[mask == 1]
        xyz_grads = StandardScaler(with_mean=False).fit_transform(xyz_grads) * scale_xyz_grads
        # hist = ff.create_distplot([xyz_grads[:, 0].tolist(), xyz_grads[:, 1].tolist(), xyz_grads[:, 2].tolist()], bin_size=0.5, group_labels=['x', 'y', 'z'])
        # st.plotly_chart(go.Figure(hist))
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            x_, y_, z_ = xyz_grads[i]
            fig.add_scatter3d(
                x=[x, x + x_],
                y=[y, y + y_],
                z=[z, z + z_],
                mode='lines',
                line=dict(
                    color="Black",
                    width=3,
                )
            )
    fig.update_layout({
        'showlegend': False,
        'scene': {
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
        }
    })
    return fig


if __name__ == '__main__':
    dset = SolidMNISTSubset(root_dir='dataset/bin', split='test')
    for i in range(5):
        graph, label, meta, image, graph_file = dset[i]

        st.plotly_chart(face_adjacency_plot(graph))

        st.plotly_chart(uv_samples_plot(*graph_to_xyz_mask(graph)))
