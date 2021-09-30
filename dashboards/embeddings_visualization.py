import os
import os.path as osp
import shutil
import sys

import numpy as np
import streamlit as st
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from streamlit.report_thread import get_report_ctx
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP


file_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(file_dir)
sys.path.append(project_root)
from st_executor import SingleThreadExecutor
from st_tensorboard import StEmbeddingProjector
from utils import solid_to_img_tensor
from upload_pos_neg import UploadExamples
from networks.models import get_abc_encoder


def main():
    executor = SingleThreadExecutor.instance()
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
        examples = executor.queue_and_block(UploadExamples, model, layers, step_files).result()
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
                grams = executor.queue_and_block(examples.grams).result()

            with st.spinner('Performing PCA...'):
                pca_grams = []
                for layer in range(7):
                    x = grams[layer].detach().cpu().numpy()
                    pca = executor.queue_and_block(PCA(n_components=min(x.shape[-1], 3, x.shape[0])).fit_transform, x).result()
                    pca_grams.append(pca)
                X = torch.from_numpy(np.concatenate(pca_grams, axis=-1))

            with st.spinner('Fitting...'):
                X_ = executor.queue_and_block(fitter.fit_transform, X).result()

            with st.spinner('Preparing Embedding Projector...'):
                log_dir = f'log_dir_{get_report_ctx().session_id}'
                shutil.rmtree(log_dir, ignore_errors=True)
                os.makedirs(osp.join(project_root, log_dir), exist_ok=True)

                img = executor.queue_and_block(get_imgs, examples).result()

                writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
                writer.add_embedding(X_, label_img=img, metadata=examples.names)
                projector = StEmbeddingProjector(osp.join(project_root, log_dir))

            projector.display()


def get_imgs(examples):
    img_tensors = []
    for solid in examples.solids:
        t = solid_to_img_tensor(solid, width=64, height=64, line_width=2)
        img_tensors.append(t)
    img = torch.stack(img_tensors)
    return img


if __name__ == '__main__':
    main()
