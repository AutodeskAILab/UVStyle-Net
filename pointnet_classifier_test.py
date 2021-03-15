import argparse
import math
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import helper
import os.path as osp
import logging
from torch.optim import lr_scheduler
import numpy as np
import sklearn.metrics as metrics
from solid_mnist import collate_with_pointclouds, SolidMNISTWithPointclouds, SolidMNISTWithPointcloudsFontSubset
from networks import pointnet

def compute_activation_stats_psnet(bg, layer, activations):
    grams = torch.matmul(activations, activations.transpose(1, 2)) / activations.shape[-1]
    triu_idx = torch.triu_indices(*grams.shape[1:])
    triu = grams[:, triu_idx[0, :], triu_idx[1, :]].flatten(start_dim=1)
    assert not triu.isnan().any()
    return triu.detach().cpu()

def log_activation_stats(bg, all_layers_activations):
    stats = {layer: compute_activation_stats_psnet(bg, layer, activations)
             for layer, activations in all_layers_activations.items()}
    return stats


def val_one_epoch(model, loader, epoch, args):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    stats = {}
    all_graph_files = []
    with torch.no_grad():
        for _, (bg, points, labels, graph_files) in enumerate(loader):
            points = points.to(args.device) 
            labels = labels.to(args.device).squeeze(-1)
            points = points.transpose(-1, 1)
            logits = model(points)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())
            for activations in [model.activations]:
                batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]
            all_graph_files += graph_files

    out_dir = 'analysis/psnet_data/solidmnist_subset'
    os.makedirs(out_dir, exist_ok=True)
    print('writing stats...')
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

    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
#     print("[Val]   Epoch {:03} Loss {:2.3f}, Acc {}".format(
#         epoch, avg_loss.item(), acc))
    return avg_loss, acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="", help='Path to saved state')
    parser.add_argument("--no-cuda", action='store_true', help='Do not use CUDA')
    parser.add_argument("--plot_confusion_matrix", action='store_true', help='Plot the confusion matrix')
    parser.add_argument("--times", type=int, default=5)
    args = parser.parse_args()

    # Load everything from state
    if len(args.state) == 0:
        raise ValueError("Expected a valid state filename")

    state = helper.load_checkpoint(args.state)
    print(state['args'])
   

    device = state['args'].device
    args.device = device

  
    test_dset = SolidMNISTWithPointcloudsFontSubset(bin_root_dir='dataset/bin', npy_root_dir='dataset/pc')
    
    test_loader = helper.get_dataloader(
        test_dset, state['args'].batch_size, train=True, collate_fn=collate_with_pointclouds)
    

    iteration = 0
    best_loss = float("inf")
    best_acc = 0

    # Train/validate
 
    best_val_acc = []

    for t in range(args.times):
        #print("Initial loss must be close to {}".format(-math.log(1.0 / test_dset.num_classes)))
        model = pointnet.PointNet(state['args'].emb_dims, test_dset.num_classes).to(device)
        
        #print("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        
        model.load_state_dict(state['model'])

        best_acc = 0.0
        print("Running experiment {}/{} times".format(t + 1, args.times))
        vloss, vacc = val_one_epoch(model, test_loader, 0, args)
      
        #print("#########################################")
        best_val_acc.append(vacc)
        print("Times {} validation accuracy: {:2.7f}".format(t, vacc))
        print("----------------------------------------------------")

    print("Best average validation accuracy: {:2.7f}+-{:2.7f}".format(np.mean(best_val_acc),
                                                                      np.std(best_val_acc)))
    print("=====================================================")


