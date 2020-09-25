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


def train_one_epoch(model, loader, optimizer, epoch, iteration, args):
    model.train()
    total_loss_array = []
    mean_acc_array = []
    train_true = []
    train_pred = []
    for _, (bg, labels) in enumerate(loader):
        iteration = iteration + 1
        optimizer.zero_grad()

        feat = bg.ndata['x'].permute(0, 2, 1).to(args.device)
        labels = labels.to(args.device).squeeze(-1)
        logits = model(bg, feat)
        #print("logits: ", logits.shape)
        #print("Label size: ", labels.shape)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        total_loss_array.append(loss.item())
        train_true.append(labels.cpu().numpy())
        preds = logits.max(dim=1)[1]
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = metrics.accuracy_score(train_true, train_pred)

    avg_loss = np.mean(total_loss_array)

    logging.info("[Train] Epoch {:03} Loss {:2.3f}, Acc {}".format(
        epoch, avg_loss.item(), acc))
    return avg_loss, acc


def val_one_epoch(model, loader, epoch, args):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for _, (bg, labels) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 2, 1).to(args.device)
            labels = labels.to(args.device).squeeze(-1)
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
    logging.info("[Val]   Epoch {:03} Loss {:2.3f}, Acc {}".format(
        epoch, avg_loss.item(), acc))
    return avg_loss, acc


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
    if args.input_channels == 'xyz_only':
        tokens.append(args.input_channels)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = parse_util.get_train_parser("UV-Net Classifier Training Script for Wires")
    # B-rep face
    parser.add_argument('--nurbs_model_type', type=str, choices=('cnn', 'wcnn'), default='cnn',
                        help='Feature extractor for NURBS surfaces')
    parser.add_argument('--nurbs_emb_dim', type=int, default=64,
                        help='Embedding dimension for NURBS feature extractor (default: 64)')
    parser.add_argument('--input_channels', type=str, choices=('xyz_only', 'xyz_normals'), default='xyz_only',
                        help='Input channels to use')
    # B-rep graph
    parser.add_argument('--brep_model_type', type=str, default='gin_grouping',
                        help='Feature extractor for B-rep face-adj graph')
    parser.add_argument('--graph_emb_dim', type=int,
                        default=128, help='Embeddings before graph pooling')
    # Classifier
    parser.add_argument('--classifier_type', type=str, choices=('linear', 'non_linear'), default='non_linear',
                        help='Classifier model')
    parser.add_argument('--final_dropout', type=float,
                        default=0.3, help='final layer dropout (default: 0.3)')
    # Dataset
    parser.add_argument('--size_percentage', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--apply_line_symmetry', type=float, default=0.3,
                        help='Probability of randomly applying a line symmetry transform on the curve grid')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    device = "cuda:" + str(args.device)
    args.device = device

    exp_name = experiment_name(args)

    # Create directories for checkpoints and logging
    log_filename = osp.join('dump', exp_name, 'log.txt')
    checkpoint_dir = osp.join('dump', exp_name, 'checkpoints')
    img_dir = osp.join('dump', exp_name, 'imgs')
    helper.create_dir(checkpoint_dir)
    helper.create_dir(img_dir)
    
    # Setup logger
    helper.setup_logging(log_filename)
    logging.info(args)
    logging.info("Experiment name: {}".format(exp_name))

    # Load datasets
    train_dset = FontWires(args.dataset_path, split='train', size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry)
    val_dset = FontWires(args.dataset_path, split='val', size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry)
    train_loader = helper.get_dataloader(
        train_dset, args.batch_size, train=True, collate_fn=collate_graphs)
    val_loader = helper.get_dataloader(
        val_dset, args.batch_size, train=False, collate_fn=collate_graphs)

    iteration = 0
    best_loss = float("inf")
    best_acc = 0

    # Train/validate
    # Arrays to store training and validation losses and accuracy
    train_losses = np.zeros((args.times, args.epochs), dtype=np.float32)
    val_losses = np.zeros((args.times, args.epochs), dtype=np.float32)
    train_acc = np.zeros((args.times, args.epochs), dtype=np.float32)
    val_acc = np.zeros((args.times, args.epochs), dtype=np.float32)
    best_val_acc = []

    for t in range(args.times):
        logging.info("Initial loss must be close to {}".format(-math.log(1.0 / train_dset.num_classes)))
        model = Model(train_dset.num_classes, args).to(device)
        logging.info("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        optimizer = helper.get_optimizer(args.optimizer, model, lr=args.lr)

        best_acc = 0.0
        logging.info("Running experiment {}/{} times".format(t + 1, args.times))

        for epoch in range(1, args.epochs + 1):
            tloss, tacc = train_one_epoch(
                model, train_loader, optimizer, epoch, iteration, args)
            train_losses[t, epoch - 1] = tloss
            train_acc[t, epoch - 1] = tacc

            vloss, vacc = val_one_epoch(model, val_loader, epoch, args)
            val_losses[t, epoch - 1] = vloss
            val_acc[t, epoch - 1] = vacc

            if vacc > best_acc:
                best_acc = vacc
                helper.save_checkpoint(osp.join(checkpoint_dir, f'best_{t}.pt'), model,
                                       optimizer, args=args)

        best_val_acc.append(best_acc)
        logging.info("Best validation accuracy: {:2.3f}".format(best_acc))
        logging.info("----------------------------------------------------")

    logging.info("Best average validation accuracy: {:2.3f}+-{:2.3f}".format(np.mean(best_val_acc),
                                                                      np.std(best_val_acc)))
    logging.info("=====================================================")

    # Plot learning curves
    import matplotlib.pyplot as plt
    import plot_utils
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim(0, args.epochs)
    ax1.set_ylim(0, -math.log(1.0 / train_dset.num_classes))
    ax2.set_xlim(0, args.epochs)
    ax2.set_ylim(0, 100)
    fig.set_size_inches(24, 8)
    train_losses_mean = np.mean(train_losses, axis=0)
    train_losses_std = np.std(train_losses, axis=0)
    val_losses_mean = np.mean(val_losses, axis=0)
    val_losses_std = np.std(val_losses, axis=0)
    plot_utils.error_curve(ax1, train_losses_mean, train_losses_std, style='band',
                            label='Training Set', c='r', facecolor='darkred')
    plot_utils.error_curve(ax1, val_losses_mean, val_losses_std, style='band',
                            label='Validation Set', c='b', facecolor='darkblue')
    ax1.legend()
    train_acc *= 100.0
    val_acc *= 100.0
    train_acc_mean = np.mean(train_acc, axis=0)
    train_acc_std = np.std(train_acc, axis=0)
    val_acc_mean = np.mean(val_acc, axis=0)
    val_acc_std = np.std(val_acc, axis=0)
    plot_utils.error_curve(ax2, train_acc_mean, train_acc_std, style='band', label='Training Set',
                            c='r', facecolor='darkred')
    plot_utils.error_curve(ax2, val_acc_mean, val_acc_std, style='band', label='Validation Set',
                            c='b', facecolor='darkblue')
    ax2.legend()
    title = " ".join(map(str, [exp_name, "Acc.: {:2.3f}%".format(best_acc * 100.0)]))
    fig.suptitle(title)
    plt.savefig(osp.join('dump', exp_name, 'imgs',
                            'learning_curves.png'), bbox_inches='tight')
    # plt.show()
