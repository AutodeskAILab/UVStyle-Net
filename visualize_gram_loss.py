import argparse

import numpy as np
import dgl
import streamlit as st
import torch

import helper
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from datasets.abcdataset import ABCDataset
from reconstruction import compute_activation_stats
from train_test_recon_pc import get_model


def weight_layers(grams, weights):
    weighted = []
    for gram, weight in zip(grams, weights):
        weighted.append(gram * weight)
    try:
        if isinstance(grams[0], torch.Tensor):
            return torch.cat(weighted, dim=-1)
    except:
        pass
    return np.concatenate(weighted, axis=-1)


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x0, x1):
        return 1 - (torch.dot(x0, x1) / (torch.norm(x0) * torch.norm(x1)))


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

    bg, feat, style_emb = compute_grams_from_model_with_grads(dgl.batch([g0, g1]), model_checkpoint, weights, device)

    loss = crit(*style_emb)
    loss.backward()
    feat_grads = []
    st.sidebar.subheader('XYZ + Normals')
    selection = [st.sidebar.checkbox(label=str(i), value=i > 2) for i in range(6)]
    idx = torch.arange(6)[selection]
    for grads in torch.split(feat.grad, bg.batch_num_nodes().tolist()):
        grads = grads[:, :, :, idx]
        if len(idx) > 1:
            combined_grads = grads.norm(dim=-1)
        else:
            combined_grads = grads
        feat_grads.append(combined_grads.flatten())
    grads = torch.split(feat.grad, bg.batch_num_nodes().tolist())
    a = uv_samples_plot(*graph_to_xyz_mask(g0),
                        sample_colors=feat_grads[0].detach().cpu(),
                        xyz_grads=grads[0][:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh0,
                        mesh_alpha=mesh_alpha)
    b = uv_samples_plot(*graph_to_xyz_mask(g1),
                        sample_colors=feat_grads[1].detach().cpu(),
                        xyz_grads=grads[1][:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh1,
                        mesh_alpha=mesh_alpha)
    return a, b


def compute_grams_from_model_with_grads(bg, model_checkpoint, weights=None, device='cpu'):
    state = helper.load_checkpoint(model_checkpoint)
    model, step = get_model(state['args'])
    model = model.to(device)
    model.load_state_dict(state['model'])
    feat = bg.ndata['x'].to(device)  # type: torch.Tensor
    feat.requires_grad = True
    in_feat = feat.permute(0, 3, 1, 2)
    model(bg.to(device), in_feat)
    activations = {}
    for acts in [model.surf_encoder.activations, model.graph_encoder.activations]:
        activations.update(acts)
    activations = {
        layer: activations for layer, activations in activations.items()
    }
    grams = [
        compute_activation_stats(bg, layer, activations[layer])
        for layer in ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']
    ]
    if weights is None:
        weights = torch.ones(len(grams))
    style_emb = weight_layers(grams, weights)
    return bg, feat, style_emb


if __name__ == '__main__':
    text = st.sidebar.text_area(label='Enter letter names (separate lines)',
                                value='3 Fillet - Part 1\n4 Chamfer - Part 1')
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
    weights_sliders = [
        st.sidebar.slider(label=str(i),
                          min_value=0.,
                          max_value=1.,
                          value=1. if i < 4 else 0.) for i in range(7)
    ]
    dset = ABCDataset(root_dir='/home/pete/brep_style/abc/bin', split='all', in_memory=False)
    model_checkpoint = '/home/pete/brep_style/grams_and_models/abc/uvnet/best.pt'

    graph_files = np.array(list(map(lambda n: n.stem, dset.graph_files)))

    names = text.split('\n')
    idx = list(map(lambda name: np.argwhere(graph_files == name).__int__(), names))
    # idx = [25460, 28541]
    g0 = dset[idx[0]]
    g1 = dset[idx[1]]

    # weights = torch.ones(7, requires_grad=True)
    torch.autograd.set_detect_anomaly(True)
    weights = torch.tensor(weights_sliders, requires_grad=True)
    a, b, = uvnet_gram_loss_vis_plot(g0, g1, weights, model_checkpoint, scale_grads=grads_slider,
                                     mesh_alpha=mesh_alpha,
                                     marker_size=marker_slider,
                                     device='cuda:0')
    st.plotly_chart(a)
    st.plotly_chart(b)
