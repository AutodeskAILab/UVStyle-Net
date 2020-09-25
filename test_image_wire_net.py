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
from networks import classifier
import parse_util
import random 

class Model(nn.Module):
    def __init__(self, num_classes, args):
        """
        Model used in this classification experiment
        """
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, 5, stride=1, padding=2),
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
        out = self.cls(x)
        return out
    
def test_one_epoch(model, loader, epoch, args):
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
    print("[Test]   Epoch {:03} Loss {:2.3f}, Acc {}".format(
        epoch, avg_loss.item(), acc))
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="", help='Path to saved state')
    parser.add_argument("--encoder_type", type=str, default="brep", help='brep or pointnet')
    parser.add_argument("--no-cuda", action='store_true', help='Do not use CUDA')
    parser.add_argument("--plot_confusion_matrix", action='store_true', help='Plot the confusion matrix')
    parser.add_argument("--seed",  default=0, help='Seed')
    parser.add_argument("--device",  default=0, help='device num')
    parser.add_argument("--img_dir",  default="", help='image directory to store images')
    parser.add_argument("--times", type=int, default=5)
    args = parser.parse_args()
    
    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load everything from state
    if len(args.state) == 0:
        raise ValueError("Expected a valid state filename")

    state = torch.load(args.state, map_location='cpu')
    print(state['args'])
   

    device = "cuda:" + str(args.device)
    #args.device = device

  
    test_dset = FontWiresWithImages(state['args'].dataset_path,  split="test", size_percentage=state['args'].size_percentage,  in_memory=True)
    
    test_loader = helper.get_dataloader(
        test_dset, 32, train=True, collate_fn=collate_graphs_with_images)
    
    # Train/validate
 
    #array_clustering_acc = []
    #array_chamfer_loss = []
    array_classification_loss = []
    for t in range(args.times):
        
        
        model = Model(26, state['args']).to(device)
            
        model.load_state_dict(state['model'])
        print("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


        #visualize(args, model, test_loader, t, device, args.img_dir)
        
        print("Running experiment {}/{} times".format(t + 1, args.times))
        #cluster_acc = cluster(args, model, test_loader, device)
        #array_clustering_acc.append(cluster_acc)
        
        test_loss, acc = test_one_epoch(model, test_loader, t, args)
        array_classification_loss.append(acc)
        
        

    print("Classification accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_classification_loss), np.std(array_classification_loss)))
    #print("Clustering accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_clustering_acc), np.std(array_clustering_acc)))
    print("=====================================================")
    
    
if __name__ == '__main__':
    main()
