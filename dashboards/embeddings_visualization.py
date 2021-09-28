import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from os import kill
from time import sleep

import numpy as np
import requests
import streamlit as st
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from streamlit.components import v1 as components
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP

file_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(file_dir)
sys.path.append(project_root)
from utils import solid_to_img_tensor
from upload_pos_neg import UploadExamples
from networks.models import get_abc_encoder

selection_html = """
<style>
#my-div
{
    width    : 1000px;
    height   : 800px;
    overflow : hidden;
    position : relative;
}
#my-iframe
{
    position : absolute;
    top      : -68px;
    left     : -313px;
    width    : 1602px;
    height   : 868px;
}
</style>
<div id="my-div">
<iframe src="http://localhost:6006/#projector" id="my-iframe" scrolling="no"></iframe>
</div>
"""


def main():
    st.set_page_config(layout='wide')
    checkpoint = osp.join(project_root, 'checkpoints', 'uvnet_abc_chkpt.pt')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_abc_encoder(checkpoint=checkpoint, device=device)
    layers = ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']

    step_files = st.sidebar.file_uploader(label='Set of B-Reps',
                                          type=['step', 'stp'],
                                          accept_multiple_files=True)
    if len(step_files) == 0:
        st.write('Upload a set of STEP files in the sidebar to get started.')
    else:
        # files are uploaded
        examples = UploadExamples(model=model,
                                  layers=layers,
                                  files=step_files)
        st.write(f'Files uploaded: {len(examples)}')

        algo = st.radio(label='Method',
                        options=['UMAP', 't-SNE', 'PCA'])

        if algo == 'PCA':
            fitter = PCA(n_components=3)
        elif algo == 'UMAP':
            umap_metric = st.selectbox(label='metric',
                                       options=['cosine', 'euclidean'])
            umap_max_val = len(examples) - 1 if umap_metric == 'cosine' else len(examples) - 2
            umap_n = st.number_input(label='n_neighbours',
                                     min_value=1,
                                     max_value=umap_max_val,
                                     value=min(15, umap_max_val))
            fitter = UMAP(n_neighbors=umap_n,
                          n_components=3,
                          metric=umap_metric)
        elif algo == 't-SNE':
            tsne_metric = st.selectbox(label='metric',
                                       options=['cosine', 'euclidean'])
            tsne_n_iter = st.number_input(label='iterations',
                                          min_value=1,
                                          max_value=10_000,
                                          value=1_000,
                                          step=1)
            tsne_perplexity = st.number_input(label='perplexity',
                                              min_value=2,
                                              max_value=100,
                                              value=5,
                                              step=1)
            tsne_lr = st.number_input(label='learning rate',
                                      min_value=0.001,
                                      max_value=1000.,
                                      value=10.,
                                      step=0.001)
            fitter = TSNE(n_components=3,
                          n_iter=tsne_n_iter,
                          metric=tsne_metric,
                          perplexity=tsne_perplexity,
                          learning_rate=tsne_lr)

        if st.button('Visualize'):
            with st.spinner('Computing Gram matrices...'):
                grams = examples.grams()

            with st.spinner('Performing PCA...'):
                pca_grams = []
                for layer in range(7):
                    x = grams[layer].detach().cpu().numpy()
                    pca = PCA(n_components=min(x.shape[-1], 3, x.shape[0])).fit_transform(x)
                    pca_grams.append(pca)
                X = torch.from_numpy(np.concatenate(pca_grams, axis=-1))

            with st.spinner('Fitting...'):
                X_ = fitter.fit_transform(X)

            with st.spinner('Preparing Embedding Projector...'):
                log_dir = 'temp_log/'
                shutil.rmtree(log_dir, ignore_errors=True)
                os.makedirs(osp.join(project_root, log_dir), exist_ok=True)

                img_tensors = []
                for solid in examples.solids:
                    t = solid_to_img_tensor(solid, width=64, height=64, line_width=2)
                    img_tensors.append(t)
                img = torch.stack(img_tensors)

                writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
                writer.add_embedding(X_, label_img=img, metadata=examples.names)

                if 'tb' in st.session_state:
                    kill(st.session_state.tb, 9)

                process = subprocess.Popen(f'bash -c "tensorboard --logdir {project_root}/temp_log"',
                                           env=os.environ,
                                           shell=True)
                st.session_state.tb = process.pid

                tensorboard_url = 'http://localhost:6006'
                success = False
                start = time.time()
                while not success:
                    try:
                        requests.get(tensorboard_url)
                        success = True
                    except Exception:
                        if time.time() - start > 10:
                            raise Exception('Cannot connect to embedding projector (timeout).')
                        sleep(0.1)
            st.text('Use left mouse button to rotate, right mouse button to pan, and scroll wheel to zoom.')
            components.html(html=selection_html,
                            width=1000,
                            height=1000)


if __name__ == '__main__':
    main()
