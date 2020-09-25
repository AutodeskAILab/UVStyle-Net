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
from font_wires import FontWiresWithImages, collate_graphs_with_images
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
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 64 x 64
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 32, 5, stride=2, padding=2),
                        nn.InstanceNorm2d(32),
                        nn.ReLU()
                     ) # 32 x 32
        self.conv3 = nn.Sequential(
                        nn.Conv2d(32, 64, 5, stride=1, padding=2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 32 x 32
        self.conv4 = nn.Sequential(
                        nn.Conv2d(64, 64, 5, stride=2, padding=2),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 16 x 16
        
        self.conv5 = nn.Sequential(
                        nn.Conv2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 8 x 8
        self.conv6 = nn.Sequential(
                        nn.Conv2d(64, 64, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU()
                     ) # 4 x 4
        
        
        self.cls = classifier.get_classifier(
            args.classifier_type, 4*4*64, num_classes, args.final_dropout)

    def forward(self, imgs):
        batch_size = imgs.size(0)
        x = self.conv1(imgs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(batch_size,-1)
        #print(x.shape)
        #raise "Err"
        out = self.cls(x)
        return out


def train_one_epoch(model, loader, optimizer, epoch, iteration, args):
    model.train()
    total_loss_array = []
    mean_acc_array = []
    train_true = []
    train_pred = []
    for _, (bg, images, labels) in enumerate(loader):
        iteration = iteration + 1
        optimizer.zero_grad()

        feat = bg.ndata['x'].permute(0, 2, 1).to(args.device)
        labels = labels.to(args.device).squeeze(-1)
        images = images.to(args.device)
        images = images.permute(0,3,1,2)
        
        logits = model(images)
        #print(logits.shape)
        #raise "err"
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
        for _, (bg, images, labels) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 2, 1).to(args.device)
            labels = labels.to(args.device).squeeze(-1)
            images = images.to(args.device)
            images = images.permute(0,3,1,2)

            logits = model(images)
      
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
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = parse_util.get_train_parser("NURBS-Net Image Classifier Training Script for Wires")
    # B-rep face
    parser.add_argument('--nurbs_model_type', type=str, choices=('cnn', 'wcnn'), default='cnn',
                        help='Feature extractor for NURBS surfaces')
    parser.add_argument('--nurbs_emb_dim', type=int, default=64,
                        help='Embedding dimension for NURBS feature extractor (default: 64)')
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
    train_dset = FontWiresWithImages(args.dataset_path,  split="train", size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry, in_memory=True)
    val_dset = FontWiresWithImages(args.dataset_path, split="val", size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry, in_memory=True)
    
    train_loader = helper.get_dataloader(
        train_dset, args.batch_size, train=True, collate_fn=collate_graphs_with_images)
    val_loader = helper.get_dataloader(
        val_dset, args.batch_size, train=False, collate_fn=collate_graphs_with_images)

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