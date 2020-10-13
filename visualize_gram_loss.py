import numpy as np
import dgl
import streamlit as st
import torch
import trimesh
from torch.nn.functional import cosine_similarity

from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from helper import load_checkpoint
from solid_mnist import SolidMNISTSubset, SolidMNISTSingleLetter, SolidMNIST
from test_classifier import Model
from analysis.util import weight_layers


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

    bg, feat, style_emb = compute_grams_from_model_with_grads(dgl.batch([g0, g1]), model_checkpoint, weights, device)

    loss = crit(*style_emb)
    loss.backward()
    normal_angles = []
    st.sidebar.subheader('XYZ + Normals')
    for feats, grads in zip(torch.split(feat, bg.batch_num_nodes().tolist()), torch.split(feat.grad, bg.batch_num_nodes().tolist())):
        normals = feats[:, :, :, 3:6].flatten(end_dim=-2)
        normal_grads = grads[:, :, :, 3:6].flatten(end_dim=-2)
        angles = torch.acos(cosine_similarity(normals, normals + normal_grads))
        normal_angles.append(angles)
    grads = torch.split(feat.grad, bg.batch_num_nodes().tolist())
    a = uv_samples_plot(*graph_to_xyz_mask(g0),
                        sample_colors=normal_angles[0],
                        xyz_grads=grads[0][:, :, :, :3].reshape([-1, 3]),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh0,
                        mesh_alpha=mesh_alpha)
    b = uv_samples_plot(*graph_to_xyz_mask(g1),
                        sample_colors=normal_angles[1],
                        xyz_grads=grads[1][:, :, :, :3].reshape([-1, 3]),
                        scale_xyz_grads=scale_grads,
                        marker_size=marker_size,
                        mesh=mesh1,
                        mesh_alpha=mesh_alpha)
    return a, b


def compute_grams_from_model_with_grads(bg, model_checkpoint, weights=None, device='cpu'):
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
    if weights is None:
        weights = torch.ones(len(grams))
    style_emb = weight_layers(grams, weights)
    return bg, feat, style_emb


if __name__ == '__main__':
    device = torch.device('cuda:0')
    mesh_path = '/home/pete/brep_style/solidmnist/mesh/test'
    model_checkpoint = '/home/pete/brep_style/grams_and_models/solidmnist/uvnet/best_0.pt'

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
    weights_sliders = [
        st.sidebar.slider(label=str(i),
                          min_value=0.,
                          max_value=1.,
                          value=1.) for i in range(7)
    ]
    dset = SolidMNIST(root_dir='dataset/bin', split='test')

    graph_files = np.array(list(map(lambda n: n.stem, dset.graph_files)))

    names = text.split('\n')
    idx = list(map(lambda name: np.argwhere(graph_files == name).__int__(), names))

    g0 = dset[idx[0]][0]
    g1 = dset[idx[1]][0]

    mesh0 = trimesh.load(f'{mesh_path}/{names[0]}.stl') if show_mesh else None
    mesh1 = trimesh.load(f'{mesh_path}/{names[1]}.stl') if show_mesh else None

    # weights = torch.ones(7, requires_grad=True)
    weights = torch.tensor(torch.tensor(weights_sliders), requires_grad=True, device=device)
    a, b, = uvnet_gram_loss_vis_plot(g0, g1, weights, model_checkpoint,
                                     device=device,
                                     scale_grads=grads_slider,
                                     mesh0=mesh0, mesh1=mesh1, mesh_alpha=mesh_alpha,
                                     marker_size=marker_slider)
    st.plotly_chart(a)
    st.plotly_chart(b)
