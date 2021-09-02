import argparse
import math
import os.path as osp

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

import helper
from networks import pointnet
from solid_mnist import collate_with_pointclouds, SolidMNISTWithPointclouds


def train_one_epoch(model, loader, optimizer, scheduler, epoch, iteration, args):
    model.train()
    total_loss_array = []
    mean_acc_array = []
    train_true = []
    train_pred = []
    #graphs, pc, label
    for _, (_, points, labels) in enumerate(loader):

        iteration = iteration + 1
        optimizer.zero_grad()

        #feat = bg.ndata['x'].permute(0, 3, 1, 2).to(args.device)
        points = points.to(args.device) 
        labels = labels.to(args.device).squeeze(-1)
        points = points.transpose(-1, 1)
        logits = model(points)
        #print("logits: ", logits.shape)
        #print("Label size: ", labels.shape)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        total_loss_array.append(loss.item())
        train_true.append(labels.cpu().numpy())
        preds = logits.max(dim=1)[1]
        train_pred.append(preds.detach().cpu().numpy())
        if iteration % 200 == True:
            avg_loss = np.mean(total_loss_array)
            train_true_ = np.concatenate(train_true)
            train_pred_ = np.concatenate(train_pred)
            acc = metrics.accuracy_score(train_true_, train_pred_)
            print("[Train] Epoch {:03}, Iteration {}, total Loss {:2.3f}, Acc {}".format(epoch, iteration, avg_loss.item(), acc))

    scheduler.step()
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = metrics.accuracy_score(train_true, train_pred)

    avg_loss = np.mean(total_loss_array)

    print("[Train] Epoch {:03} Loss {:2.3f}, Acc {}".format(
        epoch, avg_loss.item(), acc))
    
    return avg_loss, acc


def val_one_epoch(model, loader, epoch, args):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for _, (_, points, labels) in enumerate(loader):
            points = points.to(args.device) 
            labels = labels.to(args.device).squeeze(-1)
            points = points.transpose(-1, 1)
            logits = model(points)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
    print("[Val]   Epoch {:03} Loss {:2.3f}, Acc {}".format(
        epoch, avg_loss.item(), acc))
    return avg_loss, acc


def experiment_name(args, use_timestamp: bool = False, suffix='') -> str:
    """
    Create a name for the experiment from the command line arguments to the script
    :param args: Arguments parsed by argparse
    :param suffix: Suffix string to append to experiment name
    :return: Experiment name as a string
    """
    from datetime import datetime
    tokens = ["Pointnet_Cls", args.emb_dims]
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(suffix) > 0:
        tokens.append(suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = argparse.ArgumentParser("NURBS-Net classifier for solids")

    parser.add_argument('--dataset_path', type=str, default=osp.join(osp.dirname(osp.abspath(__file__)), "dataset"),
                        help='Path to the dataset root directory')
    parser.add_argument('--npy_root_dir', type=str, default=osp.join(osp.dirname(osp.abspath(__file__)), "dataset"),
                        help='Path to the dataset root directory')
    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu device to use (default: 0)')
    
    parser.add_argument('--emb_dims', type=int,
                        default=1024, help='Embeddings before set pooling')
    parser.add_argument('--final_dropout', type=float,
                        default=0.5, help='final layer dropout (default: 0.5)')
    
    # Learning
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training and validation (default: 32)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str,
                        choices=('SGD', 'Adam'), default='Adam')
    parser.add_argument('--size_percentage', type=float, default=None)
    
    # Other
    parser.add_argument("--use-timestamp", action='store_true',
                        help='Whether to use timestamp in dump files')
    parser.add_argument("--times", type=int, default=1,
                        help='Number of times to run the experiment')
    parser.add_argument("--plot_learning_curves", action='store_true',
                        help='Whether to plot and save learning curves')
    parser.add_argument("--num_points", type=int, default=1024,
                        help='number of points for pointclouds')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    print(args)

    device = "cuda:" + str(args.device)
    args.device = device

    exp_name = experiment_name(args)

    # Create directories for checkpoints and logging
    log_filename = osp.join('dump', exp_name, 'log.txt')
    checkpoint_dir = osp.join('dump', exp_name, 'checkpoints')
    img_dir = osp.join('dump', exp_name, 'imgs')
    helper.create_dir(checkpoint_dir)
    helper.create_dir(img_dir)
    print("Experiment name: {}".format(exp_name))

    train_dset = SolidMNISTWithPointclouds(bin_root_dir=args.dataset_path, npy_root_dir=args.npy_root_dir, split="train", size_percentage=args.size_percentage, num_points=args.num_points)
    val_dset = SolidMNISTWithPointclouds(bin_root_dir=args.dataset_path, npy_root_dir=args.npy_root_dir, split="val", size_percentage=args.size_percentage, num_points=args.num_points)
    train_loader = helper.get_dataloader(
        train_dset, args.batch_size, train=True, collate_fn=collate_with_pointclouds)
    val_loader = helper.get_dataloader(
        val_dset, args.batch_size, train=False, collate_fn=collate_with_pointclouds)

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
        print("Initial loss must be close to {}".format(-math.log(1.0 / train_dset.num_classes)))
        model = pointnet.PointNet(args.emb_dims, train_dset.num_classes).to(device)
        print("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        optimizer = helper.get_optimizer(args.optimizer, model, lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, 0.000001)

        best_acc = 0.0
        print("Running experiment {}/{} times".format(t + 1, args.times))

        for epoch in range(1, args.epochs + 1):
            tloss, tacc = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, iteration, args)
            train_losses[t, epoch - 1] = tloss
            train_acc[t, epoch - 1] = tacc

            vloss, vacc = val_one_epoch(model, val_loader, epoch, args)
            val_losses[t, epoch - 1] = vloss
            val_acc[t, epoch - 1] = vacc

            if vacc > best_acc:
                best_acc = vacc
                helper.save_checkpoint(osp.join(checkpoint_dir, f'best_{t}.pt'), model,
                                       optimizer, scheduler, args=args)
        print("#########################################")
        best_val_acc.append(best_acc)
        print("Best validation accuracy: {:2.3f}".format(best_acc))
        print("----------------------------------------------------")

    print("Best average validation accuracy: {:2.3f}+-{:2.3f}".format(np.mean(best_val_acc),
                                                                      np.std(best_val_acc)))
    print("=====================================================")

    # Plot learning curves if requested
    if args.plot_learning_curves:
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
        plt.savefig(osp.join('dump', exp_name,
                             'learning_curves.png'), bbox_inches='tight')
        # plt.show()

