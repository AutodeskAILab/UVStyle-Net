import os
import os.path as osp
import sys
from multiprocessing import Process

import dgl
import streamlit as st
import torch.cuda
import trimesh
from matplotlib.cm import get_cmap
from occwl.graph import face_adjacency
from occwl.io import load_step


file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from datasets.feature_pipeline import feature_extractor
from streamlit_occ_viewer import StreamlitOCCViewer
import helper
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from networks.models import UVNetSolidEncoder
from dashboards.visualize_style_loss import CosineLoss


def compute_activation_stats(bg, layer, activations):
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        if layer == 'feats':
            mask = graph_activations[:, 6, :, :].unsqueeze(1).flatten(start_dim=2)  # F x 1 x 100
            graph_activations = graph_activations[:, :6, :, :].flatten(start_dim=2)  # F x 6 x 100
            x = graph_activations * mask
            # mean = x.sum(dim=-1, keepdims=True) / mask.sum(dim=-1, keepdims=True)
            # nans_x, nans_y, nans_z = torch.where(mean.isnan())
            # mean[nans_x, nans_y, nans_z] = 0
            # x = x - mean
        elif layer[:4] == 'conv':
            x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
            # mean = x.mean(dim=-1, keepdims=True)
            # x = x - mean
        else:
            # fc and GIN layers
            # graph_activations shape: F x d x 1
            x = graph_activations.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(0)  # 1 x d x F

        x = x.permute(1, 0, 2).flatten(start_dim=1)  # x shape: d x 100F

        if layer == 'feats':
            img_size = mask.sum()
        else:
            img_size = x.shape[-1]  # img_size = 100F

        gram = torch.matmul(x, x.transpose(0, 1)) / img_size
        triu_idx = torch.triu_indices(*gram.shape)
        triu = gram[triu_idx[0, :], triu_idx[1, :]].flatten()
        assert not triu.isnan().any()
        grams.append(triu)
    return torch.stack(grams)


if __name__ == '__main__':
    checkpoint = 'uvnet_abc_chkpt.pt'

    layers = ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']

    file_a = st.sidebar.file_uploader(label='B-Rep 1')
    file_b = st.sidebar.file_uploader(label='B-Rep 2')

    visualization = st.sidebar.radio('Visualization',
                                     options=['OCC', 'Plotly'])

    if file_a is None or file_b is None:
    # if not (file_a is None or file_b is None):
        st.text('Upload your B-Rep STEP files in the sidebar to begin.')
    else:
        success = False
        try:
            with open(osp.join(project_root, 'step1.stp'), 'wb') as f_a:
                f_a.write(file_a.getvalue())
            with open(osp.join(project_root, 'step2.stp'), 'wb') as f_b:
                f_b.write(file_b.getvalue())

            files = ['step1.stp', 'step2.stp']
            process = Process(target=lambda: [load_step(osp.join(project_root, file))[0] for file in
                                              files])
            process.start()
            process.join()
            if process.exitcode != 0:
                raise Exception('Check your files are valid STEP files.')
            solids = [load_step(osp.join(project_root, file))[0] for file in
                      files]
            success = True
        except Exception as e:
            st.error(f'Error loading files: {e}')

        if success:
            if visualization == 'Plotly':
                scale_grads = st.sidebar.slider('Scale Displacement Gradients',
                                                min_value=-0.2,
                                                max_value=0.2,
                                                value=-0.1,
                                                step=0.01)
                marker_size = st.sidebar.slider('Marker Size',
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
            weights_sliders = [
                st.sidebar.slider(label=name,
                                  min_value=0.,
                                  max_value=1.,
                                  value=default,
                                  step=.01)
                for name, default in zip(layers, [1., 1., 1., 0., 0., 0., 0.])
            ]

            with st.spinner('Computing gradients...'):
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

                state = helper.load_checkpoint(osp.join(project_root, 'checkpoints', checkpoint), map_to_cpu=True)
                chkpt_args = state['args']
                model = UVNetSolidEncoder(surf_emb_dim=64,
                                          graph_emb_dim=128,
                                          ae_latent_dim=1024,
                                          device=device)

                # remove decoder weights
                encoder_state = state['model'].copy()
                for key in state['model'].keys():
                    if key.startswith('decoder'):
                        encoder_state.pop(key)

                model.load_state_dict(encoder_state)

                nx_graphs = [face_adjacency(solid) for solid in solids]
                dgl_graphs = [dgl.from_networkx(g) for g in nx_graphs]
                feats = map(feature_extractor, solids)
                feats = list(map(torch.from_numpy, feats))
                features = torch.cat(feats, dim=0).float()

                meshes = []
                for i, solid in enumerate(solids):
                    solid.triangulate_all_faces()
                    triangles, faces = solid.get_triangles()
                    mesh = trimesh.Trimesh(vertices=triangles, faces=faces)
                    mesh.apply_scale(2. / mesh.bounding_box.extents.max())
                    meshes.append(mesh)

                bg = dgl.batch(dgl_graphs)
                features.requires_grad = True

                embeddings = model(bg, features.permute(0, 3, 1, 2))

                activations = model.surf_encoder.activations
                activations.update(model.graph_encoder.activations)

                crit = CosineLoss()
                losses = []
                for layer in layers:
                    grams = compute_activation_stats(bg, layer, activations[layer])
                    losses.append(crit(*grams))
                losses = torch.stack(losses)
                weights = torch.tensor(weights_sliders)
                loss = torch.dot(weights, losses)
                loss.backward()

                gradients = features.grad
                grads = torch.split(features.grad, bg.batch_num_nodes().tolist())

            if visualization == 'Plotly':
                with st.spinner('Generating plots...'):
                    plots = []
                    for i, (graph, grad) in enumerate(zip(dgl_graphs, grads)):
                        graph.ndata['x'] = feats[i]
                        sample_plot = uv_samples_plot(*graph_to_xyz_mask(graph),
                                                      xyz_grads=grad[:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                                                      scale_xyz_grads=scale_grads,
                                                      marker_size=marker_size,
                                                      mesh=meshes[i] if show_mesh else None,
                                                      mesh_alpha=mesh_alpha)
                        plots.append(sample_plot)

                for plot in plots:
                    st.plotly_chart(plot)

            else:
                # OCC
                cmap = get_cmap('viridis')
                with st.spinner('Generating plots...'):
                    viewers = []
                    for i, solid in enumerate(solids):
                        viewer = StreamlitOCCViewer(size=(480, 480))
                        face_areas = []
                        grad = grads[i][:, :, :, :3]
                        face_levels = grad.norm(2, dim=-1, keepdim=True).mean(dim=[1, 2])
                        viewer.display_face_colormap(solid, face_levels, color_map='viridis')
                        viewers.append(viewer)

                    for viewer in viewers:
                        viewer.show()
