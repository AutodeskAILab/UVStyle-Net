import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import helper
import os.path as osp
import logging
import numpy as np
import sklearn.metrics as metrics
from font_wires import FontWires, collate_graphs
from networks import nurbs_model
from networks import brep_model
from networks import classifier
import parse_util


class Model(nn.Module):
    def __init__(self, num_classes, args):
        """
        Model used in this classification experiment
        """
        super(Model, self).__init__()
        self.nurbs_feat_ext = nurbs_model.get_nurbs_curve_model(
            nurbs_model_type=args.nurbs_model_type,
            input_channels=args.input_channels,
            output_dims=args.nurbs_emb_dim)
        self.brep_feat_ext = brep_model.get_brep_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
        self.cls = classifier.get_classifier(
            args.classifier_type, args.graph_emb_dim, num_classes, args.final_dropout)

    def forward(self, bg, feat):
        out = self.nurbs_feat_ext(feat)
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        out = self.cls(graph_emb)
        return out


def test(model, loader, device):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for _, (bg, labels) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 2, 1).to(device)
            labels = labels.to(device).squeeze(-1)
            logits = model(bg, feat)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

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
    tokens = ["CurveClassifier", args.brep_model_type, args.nurbs_model_type,
              args.classifier_type, args.graph_emb_dim, args.nurbs_emb_dim, f'linesym_{args.apply_line_symmetry}']
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = parse_util.get_test_parser("UV-Net Classifier Testing Script for Wires")
    parser.add_argument('--apply_line_symmetry', type=float, default=0.0,
                        help='Probability of randomly applying a line symmetry transform on the curve grid')
    args = parser.parse_args()
    return args



parser = parse_util.get_test_parser("UV-Net Curve Classifier Test Script")
parser.add_argument("--apply_line_symmetry", type=float, default=0.0,
                    help="Probability of applying line symmetry transformation to u-domain")
args = parser.parse_args()

# Load everything from state
if len(args.state) == 0:
    raise ValueError("Expected a valid state filename")

state = helper.load_checkpoint(args.state)
print('Args used during training:\n', state['args'])

# Load dataset
test_dset = FontWires(state['args'].dataset_path, split="test", apply_line_symmetry=args.apply_line_symmetry)
test_loader = helper.get_dataloader(
    test_dset, state['args'].batch_size, train=False, collate_fn=collate_graphs)

# Device for training/testing
device = torch.device("cuda" if not args.no_cuda else "cpu")

# Create model and load weights
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
plt.savefig(osp.join('dump', exp_name, 'imgs', 'confusion_matrix.png'), bbox_inches='tight')
#plt.show()
