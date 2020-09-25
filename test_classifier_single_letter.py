import torch
import os

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb

import helper
import parse_util
from networks import brep_model
from networks import classifier
from networks import nurbs_model
from solid_mnist import collate_single_letter, SolidMNISTSingleLetter


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
        out = self.cls(graph_emb)
        return out


def compute_activation_stats(bg, layer, activations):
    means = []
    sigmas = []
    covs = []
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        # F = num faces
        # d = num filters/dimensions
        # graph_activations shape: F x d x 10 x 10
        x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
        x = torch.cat(list(x), dim=-1)  # x shape: d x 100F
        inorm = torch.nn.InstanceNorm1d(x.shape[0])
        x = inorm(x.unsqueeze(0)).squeeze()
        img_size = x.shape[-1]  # img_size = 100F
        gram = torch.matmul(x, x.transpose(0, 1)) / img_size
        mean = torch.mean(x, 1, keepdim=True)  # mean shape: d x 1
        x = (x - mean)  # x shape: d x 100F
        cov = torch.matmul(x, x.transpose(0, 1)) / (img_size - 1)  # cov shape: d x d
        means.append(mean.flatten())
        covs.append(cov.flatten())
        sigmas.append(cov.diag())
        grams.append(gram.flatten())
    return torch.stack(means).detach().cpu(),\
           torch.stack(sigmas).detach().cpu(),\
           torch.stack(covs).detach().cpu(),\
           torch.stack(grams).detach().cpu()


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
    with torch.no_grad():
        for batch, (bg, labels, graph_files) in enumerate(loader):
            print('batch: ', batch)
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
            labels = labels.to(device).squeeze(-1)
            logits = model(bg, feat)

            for activations in [model.nurbs_activations, model.gnn_activations]:
                batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]

            all_graph_files += graph_files
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

    print('writing stats...')
    all_stats = {}
    for layer, layer_stats in stats.items():
        mean, sigma, cov, gram = zip(*layer_stats)
        all_stats[layer] = {
            'gram': torch.cat(gram),
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


if __name__ == '__main__':
    out_dir = 'analysis/uvnet_data/solidmnist_single_letter'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
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
    test_dset = SolidMNISTSingleLetter('dataset/bin', split="test", target_letter='G')

    test_loader = helper.get_dataloader(
        test_dset, state['args'].batch_size, train=False, collate_fn=collate_single_letter)

    # Device for training/testing
    device = torch.device("cuda" if not args.no_cuda else "cpu")

    # Create model and load weights
    state['args'].input_channels = 'xyz_normals'
    model = Model(test_dset.num_classes, state['args']).to(device)
    model.load_state_dict(state['model'])

    vloss, vacc, pred, true = test(model, test_loader, device)

    print("Test accuracy: {:2.3f}".format(vacc * 100.0))
    print("=====================================================")
