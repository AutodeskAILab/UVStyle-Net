import os
import os.path as osp
import random
import string
import sys
import zipfile
from abc import abstractmethod
from io import BytesIO

import dgl
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch.cuda
from sklearn.decomposition import PCA
from streamlit.report_thread import get_report_ctx

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from st_executor import SingleThreadExecutor, ManyThreadExecutor
from few_shot import optimize
from networks.models import get_abc_encoder
from utils import Grams, solid_from_file, compute_activation_stats, graphs_and_feats


class Examples:
    @abstractmethod
    def grams(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class UploadExamples(Examples):

    def __init__(self, model, layers, files) -> None:
        super().__init__()
        self._model = model
        self._layers = layers
        self._solids = []
        self._names = []

        step_files = []
        for file in files:
            if file.name[-4:] == '.zip':
                try:
                    z = zipfile.ZipFile(file)
                except Exception as e:
                    st.error(f'Cannot extract {file.name}: {e}')
                    continue
                with st.spinner(f'Extracting {file.name}...'):
                    progress = st.progress(0)
                    for i, f in enumerate(z.infolist()):
                        if f.filename[-1] == '/' or f.filename.split('/')[-1][0] == '.':
                            progress.progress((i + 1) / len(z.infolist()))
                            continue
                        try:
                            extracted = z.read(f)
                            b = BytesIO(extracted)
                            b.name = f.filename
                            step_files.append(b)
                        except Exception as e:
                            st.error(f'Cannot extract {f.filename}: {e}')
                        progress.progress((i+1) / len(z.infolist()))
                    progress.empty()

            else:
                step_files.append(file)

        with st.spinner('Processing STEP files...'):
            progress = st.progress(0)
            for i, file in enumerate(step_files):
                try:
                    solid = solid_from_file(file, temp_name=''.join(
                        random.choice(string.ascii_letters + string.digits) for _ in range(50)))
                    self._solids.append(solid)
                    self._names.append(file.name)
                except Exception as e:
                    st.error(f'Error loading \'{file.name}\': {e}')
                finally:
                    progress.progress((i + 1) / len(step_files))
            progress.empty()

    @property
    def solids(self):
        return self._solids

    @property
    def names(self):
        return self._names

    def grams(self):
        nx_graphs, dgl_graphs, feats = graphs_and_feats(self._solids)
        features = torch.cat(feats, dim=0).float()
        bg = dgl.batch(dgl_graphs)

        self._model.eval()
        _ = self._model(bg, features.permute(0, 3, 1, 2))

        activations = self._model.surf_encoder.activations
        activations.update(self._model.graph_encoder.activations)

        grams = []
        for layer in self._layers:
            gram = compute_activation_stats(bg, layer, activations[layer])
            grams.append(gram)

        return grams

    def __len__(self):
        return len(self._solids)


class RandomExamples(Examples):

    def __init__(self, grams_root, num) -> None:
        super().__init__()
        self._grams = Grams(grams_root)
        self._idx = list(np.random.choice(len(self._grams.graph_files), num, replace=False))

    def grams(self):
        return [
            torch.from_numpy(layer_grams[self._idx])
            for layer, layer_grams in enumerate(self._grams.grams[:7])
        ]

    def __len__(self):
        return len(self._idx)


def concat_pca_grams(pos_grams, neg_grams):
    grams = []
    for layer in range(7):
        X = torch.cat([pos_grams[layer], neg_grams[layer]], dim=0).detach().cpu().numpy()
        pca = PCA(n_components=min(X.shape[-1], 70, X.shape[0])).fit_transform(X)
        grams.append(pca)
    return grams


def main():
    single_executor = SingleThreadExecutor.instance()  # type: SingleThreadExecutor
    many_executor = ManyThreadExecutor.instance()  # type: ManyThreadExecutor
    checkpoint = osp.join(project_root, 'checkpoints', 'uvnet_abc_chkpt.pt')
    grams_root = osp.join(project_root, 'data', 'ABC', 'uvnet_grams', 'all')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    layers = ['feats', 'conv1', 'conv2', 'conv3', 'fc', 'GIN_1', 'GIN_2']

    model = get_abc_encoder(checkpoint, device)

    positive_files = st.sidebar.file_uploader(label='Positive Examples',
                                              type=['step', 'stp', 'zip'],
                                              accept_multiple_files=True)
    # with open('/Users/meltzep/step_files.zip', 'rb') as f:
    #     b = f.read()
    # with open(osp.join(project_root, 'pos_step_1.stp'), 'rb') as f:
    #     b2 = f.read()
    # positive_files = [UploadedFile(UploadedFileRec(0, 'step_files.zip', 'step', b))]

    negatives_option = st.sidebar.radio(label='',
                                        options=['Random Negatives', 'Upload Negatives'])

    if negatives_option == 'Upload Negatives':
        negative_files = st.sidebar.file_uploader(label='Negative Examples',
                                                  type=['step', 'stp', 'zip'],
                                                  accept_multiple_files=True)
    else:
        # random negatives from ABC
        num_negatives = st.sidebar.number_input('Number of Negatives',
                                                min_value=0,
                                                max_value=10_000,
                                                value=100,
                                                step=1)
    positives = None
    if len(positive_files) > 0:
        positives = many_executor.queue_and_block(lambda: UploadExamples(model=model,
                                                                         layers=layers,
                                                                         files=positive_files),
                                                  ctx=get_report_ctx()).result()
    negatives = None
    if negatives_option == 'Upload Negatives':
        # noinspection PyUnboundLocalVariable
        if len(negative_files) > 0:
            negatives = many_executor.queue_and_block(lambda: UploadExamples(model=model,
                                                                             layers=layers,
                                                                             files=negative_files),
                                                      ctx=get_report_ctx()).result()
    else:
        # noinspection PyUnboundLocalVariable
        negatives = many_executor.queue_and_block(lambda: RandomExamples(grams_root=grams_root,
                                                                         num=num_negatives)).result()

    if not (positives and negatives):
        st.text('Upload your B-Rep STEP files in the sidebar to begin.')
    else:
        # files are uploaded
        st.write(f'Number of Positives: {len(positives)}')
        st.write(f'Number of Negatives: {len(negatives)}')

        if st.button(label='Optimize'):
            with st.spinner('Computing Gram matrices'):
                pos_grams = many_executor.queue_and_block(positives.grams).result()
                neg_grams = many_executor.queue_and_block(negatives.grams).result()

            with st.spinner('Performing PCA'):
                grams = many_executor.queue_and_block(concat_pca_grams, pos_grams, neg_grams).result()

            with st.spinner('Optimizing'):
                weights = single_executor.queue_and_block(
                    lambda: optimize(positive_idx=np.arange(len(positives) + len(negatives))[:len(positives)],
                                     negative_idx=np.arange(len(positives) + len(negatives))[len(positives):],
                                     grams=grams,
                                     metric='cosine')).result()

            st.subheader('Optimal Weights')
            df = pd.DataFrame()
            df['Layer'] = layers
            df['Weight'] = weights
            fig = px.bar(data_frame=df,
                         x='Layer',
                         y='Weight')
            st.plotly_chart(fig)
            st.write(df)


if __name__ == '__main__':
    main()
