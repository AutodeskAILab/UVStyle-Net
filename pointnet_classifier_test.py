import argparse
import math
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
from solid_mnist import collate_with_pointclouds, SolidMNISTWithPointclouds
from networks import pointnet




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

  
    test_dset = SolidMNISTWithPointclouds(bin_root_dir=state['args'].dataset_path, npy_root_dir=state['args'].npy_root_dir, train="test", size_percentage=state['args'].size_percentage, num_points=state['args'].num_points)
    
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


