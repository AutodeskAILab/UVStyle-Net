import os
import sys

import dgl
import numpy as np
import streamlit as st
import torch
import trimesh

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from helper import load_checkpoint
from datasets.solid_mnist import SolidMNIST
from test_classifier import Model


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x0, x1):
        return 1 - (torch.dot(x0, x1) / (torch.norm(x0) * torch.norm(x1)))


def compute_grams(activations, bg):
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
        x = torch.cat(list(x), dim=-1)  # x shape: d x 100F
        inorm = torch.nn.InstanceNorm1d(x.shape[0])
        x = inorm(x.unsqueeze(0)).squeeze()
        img_size = x.shape[-1]  # img_size = 100F
        gram = torch.matmul(x, x.transpose(0, 1)) / img_size
        grams.append(gram.flatten())
    return torch.stack(grams)


def uvnet_gram_loss_vis_plot(g0, g1, weights,
                             model_checkpoint,
                             device='cpu',
                             metric='cosine',
                             scale_grads=-0.05,
                             marker_size=4,
                             mesh0=None,
                             mesh1=None,
                             mesh_alpha=1.):
    if metric == 'cosine':
        crit = CosineLoss()
    elif metric == 'euclidean':
        crit = torch.nn.MSELoss()
    else:
        raise Exception('metric must be \'cosine\' or \'euclidean\'')

    bg, feat, grams = compute_grams_from_model_with_grads(dgl.batch([g0, g1]), model_checkpoint, device)

    losses = [weights[i] * crit(*gram) for i, gram in enumerate(grams)]
    loss = sum(losses)
    loss.backward()
    feat_grads = []
    for grads in torch.split(feat.grad, bg.batch_num_nodes().tolist()):
        grads = grads[:, :, :, :3]
        if len(idx) > 1:
            combined_grads = grads.norm(dim=-1)
        else:
            combined_grads = grads
        feat_grads.append(combined_grads.flatten())
    grads = torch.split(feat.grad, bg.batch_num_nodes().tolist())
    a = uv_samples_plot(*graph_to_xyz_mask(g0),
                        xyz_grads=grads[0][:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh0,
                        mesh_alpha=mesh_alpha)
    b = uv_samples_plot(*graph_to_xyz_mask(g1),
                        xyz_grads=grads[1][:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh1,
                        mesh_alpha=mesh_alpha)
    return a, b


def compute_grams_from_model_with_grads(bg, model_checkpoint, device='cpu'):
    state = load_checkpoint(model_checkpoint, map_to_cpu=device == 'cpu')
    state['args'].input_channels = 'xyz_normals'
    model = Model(26, state['args']).to(device)
    model.load_state_dict(state['model'])
    feat = bg.ndata['x'].to(device)
    feat.requires_grad = True
    in_feat = feat.permute(0, 3, 1, 2)
    model(bg.to(device), in_feat)
    activations = {}
    for acts in [model.nurbs_activations, model.gnn_activations]:
        activations.update(acts)
    activations = {
        layer: activations for layer, activations in activations.items()
    }
    grams = [
        compute_grams(activations[layer], bg)
        for layer in ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']
    ]
    return bg, feat, grams


if __name__ == '__main__':
    mesh_path = os.path.join(project_root, 'data', 'SolidLETTERS', 'mesh', 'test')
    text = st.sidebar.text_area(label='Enter letter names (separate lines)',
                                value='c_Viaoda Libre_lower\ns_Aldrich_upper')
    st.sidebar.subheader('Options')
    grads_slider = st.sidebar.slider('Scale Displacement Gradients',
                                     min_value=-0.2,
                                     max_value=0.2,
                                     value=-0.1,
                                     step=0.01)
    marker_slider = st.sidebar.slider('Marker Size',
                                      min_value=0,
                                      max_value=10,
                                      value=4,
                                      step=1)
    show_mesh = st.sidebar.checkbox('Show Mesh',
                                    value=True)
    mesh_alpha = st.sidebar.slider('Mesh Alpha',
                                   min_value=0.,
                                   max_value=1.,
                                   step=0.1,
                                   value=0.7)
    st.sidebar.subheader('Layer Weights')
    default_weights = [1., 1., 1., 1., 0., 0., 0.]
    weights_sliders = [
        st.sidebar.slider(label=str(i),
                          min_value=0.,
                          max_value=1.,
                          value=w) for i, w in enumerate(default_weights)
    ]
    dset = SolidMNIST(root_dir=os.path.join(project_root, 'data', 'SolidLETTERS', 'bin'), split='test')
    model_checkpoint = os.path.join(project_root, 'checkpoints', 'uvnet_solidletters_chkpt.pt')

    graph_files = np.array(list(map(lambda n: n.stem, dset.graph_files)))

    names = text.split('\n')
    idx = list(map(lambda name: np.argwhere(graph_files == name).__int__(), names))

    g0 = dset[idx[0]][0]
    g1 = dset[idx[1]][0]

    mesh0 = trimesh.load(f'{mesh_path}/{names[0]}.stl') if show_mesh else None
    mesh1 = trimesh.load(f'{mesh_path}/{names[1]}.stl') if show_mesh else None

    torch.autograd.set_detect_anomaly(True)
    weights = torch.tensor(weights_sliders, requires_grad=True)
    a, b, = uvnet_gram_loss_vis_plot(g0, g1, weights, model_checkpoint, scale_grads=grads_slider,
                                     mesh0=mesh0, mesh1=mesh1, mesh_alpha=mesh_alpha,
                                     marker_size=marker_slider,
                                     device='cuda:0')
    st.plotly_chart(a)
    st.plotly_chart(b)
