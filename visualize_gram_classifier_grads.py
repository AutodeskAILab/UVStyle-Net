import os
import pickle

import dgl
import streamlit as st
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder

from graph_plotter import uv_samples_plot
from solid_mnist import SolidMNISTSubset
from visualize_gram_loss import compute_grams_from_model_with_grads


def fit_classifier(X, y):
    cache_file = 'cls_cache.pickle'
    if os.path.exists(cache_file):
        print('loading cached classifier...')
        with open(cache_file, 'rb') as file:
            coefs, intercepts = pickle.load(file)
    else:
        print('fitting classifier...')
        cls = LogisticRegressionCV(max_iter=1000,
                                   multi_class='multinomial')
        cls.fit(X, y)
        coefs = torch.tensor(cls.coef_)
        intercepts = torch.tensor(cls.intercept_)
        with open(cache_file, 'wb') as file:
            pickle.dump((coefs, intercepts), file)
    return coefs, intercepts


if __name__ == '__main__':
    model_checkpoint = 'dump/Classifier.gin_grouping.cnn.mask_channel.area_channel_False.non_linear.128.64.squaresym_0.3/checkpoints/best_0.pt'

    dset = SolidMNISTSubset('dataset/bin', split='test')
    graphs, _, _, _, names = zip(*dset)
    fonts = list(map(lambda n: n[2:-10], names))
    labels = LabelEncoder().fit_transform(fonts)

    print('computing grams...')
    bg, feat, style_emb = compute_grams_from_model_with_grads(bg=dgl.batch(graphs),
                                                              model_checkpoint=model_checkpoint,
                                                              weights=None,
                                                              device='cpu')

    coefs, intercepts = fit_classifier(style_emb.detach().numpy(), labels)

    print('computing gradients...')
    crit = torch.nn.CrossEntropyLoss()
    y = style_emb.mm(coefs.transpose(0, 1).float())

    split_feats = torch.split(feat.detach(), bg.batch_num_nodes)

    for i in range(max(labels) + 1):
        # select first instance from this class
        for j in range(2):
            idx = torch.arange(len(labels))[labels == i][j]
            xyz = split_feats[idx][:, :, :, :3].reshape(-1, 3)
            mask = split_feats[idx][:, :, :, 6].reshape(-1)
            logit = y[idx][labels[idx]]
            logit.backward(retain_graph=True)
            split_grads = torch.split(feat.grad, bg.batch_num_nodes)
            xyz_grads = split_grads[idx][:, :, :, :3].reshape(-1, 3)
            normal_grads = split_grads[idx][:, :, :, 3:6].reshape(-1, 3)
            colors = normal_grads.norm(dim=-1)

            fig = uv_samples_plot(xyz=xyz,
                                  mask=mask,
                                  sample_colors=colors,
                                  xyz_grads=xyz_grads)
            st.plotly_chart(fig)
