import argparse
import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import helper
import os
import os.path as osp
import logging
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from solid_mnist import my_collate, SolidMNIST, SolidMNISTSubset
from networks import nurbs_model
from networks import brep_model
from networks import classifier
import parse_util
import torch.utils.tensorboard as tb


class Model(nn.Module):
    def __init__(self, num_classes, args):
        """
        Model used in this classification experiment
        """
        super(Model, self).__init__()
        self.nurbs_feat_ext = nurbs_model.get_nurbs_model(
            nurbs_model_type=args.nurbs_model_type,
            output_dims=args.nurbs_emb_dim,
            mask_mode=args.mask_mode,
            area_as_channel=args.area_as_channel,
            input_channels=args.input_channels)
        self.brep_feat_ext = brep_model.get_brep_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
        self.cls = classifier.get_classifier(
            args.classifier_type, args.graph_emb_dim, num_classes, args.final_dropout)
        self.nurbs_activations = None
        self.gnn_activations = None

    def forward(self, bg, feat):
        out = self.nurbs_feat_ext(feat)
        self.nurbs_activations = self.nurbs_feat_ext.activations
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        self.gnn_activations = self.brep_feat_ext.activations
        out, emb = self.cls(graph_emb)
        return out, emb


def compute_activation_stats(bg, layer, activations):
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        if layer == 'feats':
            mask = graph_activations[:, 6, :, :].unsqueeze(1).flatten(start_dim=2)  # F x 1 x 100
            graph_activations = graph_activations[:, :6, :, :].flatten(start_dim=2)  # F x 6 x 100
            masked_activations = graph_activations * mask
            N = mask.sum(dim=-1)  # F x 1
            mean = masked_activations.sum(dim=-1) / N  # F x 6

            # handle faces that are completely masked (contain 0 samples)
            nans_x, nans_y = torch.where(mean.isnan())
            mean[nans_x, nans_y] = 0

            x_sub_mean = masked_activations - mean[:, :, None]  # F x 6 x 100
            var = torch.pow(x_sub_mean, 2).sum(dim=-1) / N  # F x 6
            std = torch.sqrt(var)  # F x 6

            nans_x, nans_y = torch.where(std.isnan())
            std[nans_x, nans_y] = 0

            epsilon = 1e-5
            x = ((graph_activations - mean[:, :, None]) / (std[:, :, None] + epsilon)) * mask  # F x 6 x 100
        elif layer[:4] == 'conv':
            x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
            # inorm is per face
            inorm = torch.nn.InstanceNorm1d(x.shape[1])
            x = inorm(x)
        else:
            # fc and GIN layers
            # graph_activations shape: F x d x 1
            x = graph_activations.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(0)

            # inorm is per solid
            inorm = torch.nn.InstanceNorm1d(x.shape[1])
            x = inorm(x)
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
    return torch.stack(grams).detach().cpu()


def log_activation_stats(bg, all_layers_activations):
    stats = {layer: compute_activation_stats(bg, layer, activations)
             for layer, activations in all_layers_activations.items()}
    return stats


def test(model, loader, device):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    stats = {}
    all_graph_files = []
    content_embeddings = []

    with torch.no_grad():
        for batch, (bg, labels, _, _, graph_files) in enumerate(loader):
            print('batch: ', batch)
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
            labels = labels.to(device).squeeze(-1)
            bg = bg.to(device)
            logits, emb = model(bg, feat)
            for activations in [model.nurbs_activations, model.gnn_activations]:
                batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]
            all_graph_files += graph_files

            content_embeddings.append(emb.detach().cpu().numpy())

            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

    print('writing stats...')
    content_embeddings = np.concatenate(content_embeddings, axis=0)
    np.save(out_dir + '/content_embeddings', content_embeddings)
    all_stats = {}
    for layer, layer_stats in stats.items():
        # gram = zip(*layer_stats)
        all_stats[layer] = {
            'gram': torch.cat(layer_stats),
        }

    for i, (layer, layer_stats) in enumerate(all_stats.items()):
        grams = layer_stats['gram'].numpy()
        np.save(out_dir + f'/{i}_{layer}_grams', grams)

    all_graph_files = list(map(lambda file: file.split('/')[-1], all_graph_files))
    pd.DataFrame(all_graph_files).to_csv(out_dir + '/graph_files.txt', index=False, header=None)
    print('done writing stats')

    print('calc metrics...')
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
    return avg_loss, acc, pred, true


def experiment_name(args) -> str:
    """
    Create a name for the experiment from the command line arguments to the script
    :param args: Arguments parsed by argparse
    :return: Experiment name as a string
    """
    from datetime import datetime
    tokens = ["Classifier", args.brep_model_type, args.nurbs_model_type, "mask_" + args.mask_mode,
              "area_channel_" + str(args.area_as_channel),
              args.classifier_type, args.graph_emb_dim, args.nurbs_emb_dim, f'squaresym_{args.apply_square_symmetry}']
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


if __name__ == '__main__':
    out_dir = 'analysis/uvnet_data/solidmnist_all_fnorm'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    parser = parse_util.get_test_parser("UV-Net Classifier Testing Script for Solids")
    parser.add_argument("--apply_square_symmetry", type=float, default=0.0,
                        help="Probability of applying square symmetry transformation to uv-domain")
    args = parser.parse_args()

    writer = tb.SummaryWriter()

    # Load everything from state
    if len(args.state) == 0:
        raise ValueError("Expected a valid state filename")

    state = helper.load_checkpoint(args.state, map_to_cpu=args.no_cuda)
    print('Args used during training:\n', state['args'])

    # Load dataset
    test_dset = SolidMNIST('dataset/bin', split="test")

    test_loader = helper.get_dataloader(
        test_dset, state['args'].batch_size, train=False, collate_fn=my_collate)

    # Device for training/testing
    device = torch.device("cuda:0" if not args.no_cuda else "cpu")

    # Create model and load weights
    state['args'].input_channels = 'xyz_normals'

    model = Model(test_dset.num_classes, state['args']).to(device)
    model.load_state_dict(state['model'])

    vloss, vacc, pred, true = test(model, test_loader, device)

    print("Test accuracy: {:2.3f}".format(vacc * 100.0))
    print("=====================================================")

    # Plot confusion matrix if requested
    import matplotlib.pyplot as plt
    import plot_utils
    import string

    classes = list(string.ascii_lowercase)[:test_dset.num_classes]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 16)
    exp_name = experiment_name(state['args'])
    plot_utils.confusion_matrix(ax, true, pred, title=exp_name, classes=classes)
    img_dir = osp.join('dump', exp_name, 'imgs')
    if not osp.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(osp.join('dump', exp_name, 'imgs', 'confusion_matrix.png'), bbox_inches='tight')
    # plt.show()
    print(writer.log_dir)
