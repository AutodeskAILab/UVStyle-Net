import os
import os.path as osp
import sys

import dgl
import streamlit as st
import torch.cuda
import trimesh
from streamlit.report_thread import get_report_ctx
from streamlit.uploaded_file_manager import UploadedFile, UploadedFileRec

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import compute_activation_stats, graphs_and_feats
from dashboards.st_executor import ManyThreadExecutor
from dashboards.st_executor import _StExecutor
from dashboards.upload_pos_neg import UploadExamples
from networks.models import get_abc_encoder
from streamlit_occ_viewer import StreamlitOCCViewer
from graph_plotter import uv_samples_plot, graph_to_xyz_mask
from dashboards.visualize_style_loss import CosineLoss


def get_meshes(solids):
    meshes = []
    for i, solid in enumerate(solids):
        solid.triangulate_all_faces()
        triangles, faces = solid.get_triangles()
        mesh = trimesh.Trimesh(vertices=triangles, faces=faces)
        mesh.apply_scale(2. / mesh.bounding_box.extents.max())
        meshes.append(mesh)
    return meshes


def get_occwl_viewers(grads, solids):
    viewers = []
    for i, solid in enumerate(solids):
        viewer = StreamlitOCCViewer(size=(480, 480))
        face_areas = []
        grad = grads[i][:, :, :, :3]
        face_levels = grad.norm(1, dim=-1, keepdim=True).mean(dim=[1, 2])
        viewer.display_face_colormap(solid, face_levels, color_map='viridis')
        viewers.append(viewer)
    return viewers


def get_plotly_plots(dgl_graphs, feats, grads, marker_size, mesh_alpha, meshes, scale_grads, show_mesh):
    plots = []
    for i, (graph, grad) in enumerate(zip(dgl_graphs, grads)):
        graph.ndata['x'] = feats[i]
        # noinspection PyUnboundLocalVariable
        sample_plot = uv_samples_plot(*graph_to_xyz_mask(graph),
                                      xyz_grads=grad[:, :, :, :3].reshape([-1, 3]).detach().cpu(),
                                      scale_xyz_grads=scale_grads,
                                      marker_size=marker_size,
                                      mesh=meshes[i] if show_mesh else None,
                                      mesh_alpha=mesh_alpha)
        plots.append(sample_plot)
    return plots


def get_loss(activations, bg, layers, weights_sliders):
    crit = CosineLoss()
    losses = []
    for layer in layers:
        grams = compute_activation_stats(bg, layer, activations[layer])
        losses.append(crit(*grams))
    losses = torch.stack(losses)
    weights = torch.tensor(weights_sliders)
    loss = torch.dot(weights, losses)
    return loss


def get_activations(bg, features, model):
    model.eval()
    _ = model(bg, features.permute(0, 3, 1, 2))
    activations = model.surf_encoder.activations
    activations.update(model.graph_encoder.activations)
    return activations


def main():
    executor = ManyThreadExecutor.instance()  # type: _StExecutor
    checkpoint = osp.join(project_root, 'checkpoints', 'uvnet_abc_chkpt.pt')

    layers = ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']

    file_a = st.sidebar.file_uploader(label='B-Rep 1')
    file_b = st.sidebar.file_uploader(label='B-Rep 2')
    # with open('/Users/meltzep/step_files/test_shape_a.step', 'rb') as f:
    #     file_a = UploadedFile(UploadedFileRec(0, 'a', 'step', f.read()))
    # with open('/Users/meltzep/step_files/test_shape_c.step', 'rb') as f:
    #     file_b = UploadedFile(UploadedFileRec(0, 'b', 'step', f.read()))

    visualization = st.sidebar.radio('Visualization',
                                     options=['OCC', 'Plotly'])

    if file_a is None or file_b is None:
        st.text('Upload your B-Rep STEP files in the sidebar to begin.')
    else:
        success = False
        try:
            with st.spinner('Processing STEP files...'):
                solids = executor.queue_and_block(lambda files: UploadExamples(None, None, files=files).solids,
                                                  [file_a, file_b], ctx=get_report_ctx()).result()
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

                model = get_abc_encoder(checkpoint=checkpoint, device=device)

                nx_graphs, dgl_graphs, feats = executor.queue_and_block(graphs_and_feats, solids).result()
                features = torch.cat(feats, dim=0).float()

                meshes = executor.queue_and_block(get_meshes, solids).result()

                bg = dgl.batch(dgl_graphs)
                features.requires_grad = True

                activations = executor.queue_and_block(get_activations, bg, features, model).result()

                loss = executor.queue_and_block(get_loss, activations, bg, layers, weights_sliders).result()
                executor.queue_and_block(loss.backward).result()
                st.text(f'Total loss for selected layers: {loss.__float__()}')

                grads = torch.split(features.grad, bg.batch_num_nodes().tolist())
            # visualization = 'Plotly'
            # scale_grads = -0.1
            # marker_size = 4
            # show_mesh = False
            # mesh_alpha = .4
            if visualization == 'Plotly':
                with st.spinner('Generating plots...'):
                    # noinspection PyUnboundLocalVariable
                    plots = executor.queue_and_block(get_plotly_plots,
                                                     dgl_graphs, feats, grads, marker_size, mesh_alpha, meshes,
                                                     scale_grads, show_mesh).result()
                    for plot in plots:
                        st.plotly_chart(plot)

            else:
                # OCC
                st.image(image=osp.join(project_root, 'dashboards', 'color_scale.png'),
                         caption='Scale')
                with st.spinner('Generating plots...'):
                    viewers = executor.queue_and_block(get_occwl_viewers, grads, solids).result()
                    for viewer in viewers:
                        viewer.show()


if __name__ == '__main__':
    main()
