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
from networks import image_encoder
from networks import decoder 
import parse_util
from torch import distributions as dist
import random
from torch.optim import lr_scheduler
from PIL import Image, ImageOps
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder_type = args.encoder_type
        if args.encoder_type == "image":
            self.image_encoder = image_encoder.Image_Encoder(args)
            self.proj_layer = nn.Linear(1024, args.latent_dim)
        else:
            self.nurbs_feat_ext = nurbs_model.get_nurbs_curve_model(
                nurbs_model_type=args.nurbs_model_type,
                output_dims=args.nurbs_emb_dim)
            self.brep_feat_ext = brep_model.get_brep_model(
                args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
            self.proj_layer = nn.Linear(args.graph_emb_dim, args.latent_dim)
        
        self.decoder = decoder.ImageDecoder(args.latent_dim)

    def forward(self,  bg, feat, imgs):
        batch_size = imgs.size(0)
        if self.encoder_type == "image":
            x = self.image_encoder(imgs)
        else:
            out = self.nurbs_feat_ext(feat)
            node_emb, x = self.brep_feat_ext(bg, out)
        
        #print(x.shape)
        embedding = self.proj_layer(x)
        
        x = self.decoder(embedding)
            
        return x, embedding
    
    
def val_image_loss(model, loader, epoch, device): 
    model.eval()
    total_loss_array = []
    with torch.no_grad():
        for _, (bg, images, labels) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 2, 1).to(device)
            images = images.to(device) 
            labels = labels.to(device).squeeze(-1)
            images = images.permute(0,3,1,2)
            
            pred_out, embedding = model(bg, feat, images)
        
            loss = F.binary_cross_entropy_with_logits(pred_out, images)
            
            #loss = chamfer_dist(points, pred_out) * 1000
            total_loss_array.append(loss.item())
    avg_loss = np.mean(total_loss_array)
    print("[Val] Epoch {:03}  Loss {:2.3f}".format(epoch, avg_loss.item()))
    print("#########################################")
    return avg_loss  

def get_embedding(args, model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for _, (bg, images, label) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 2, 1).to(device)
            images = images.to(device) 
            label = label.to(device).squeeze(-1)
            images = images.permute(0,3,1,2)
            
            pred_out, embedding = model(bg, feat, images)
            
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    #embeddings = torch.tensor(embeddings, dtype=torch.float)
    #labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels

def cluster(args, model, loader, device):
    model.eval()
    embeddings, labels = get_embedding(args, model, loader, device)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=100)
    kmeans.fit(embeddings)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(np.squeeze(labels), pred_labels)
    print("[Val] NMI score {}".format(score))
    return score

def visualize(args, model, loader, epoch, device, img_dir):
    model.train()
    total_loss_array = []
    #print(img_dir)
    for _, (bg, images, labels) in enumerate(loader):
        feat = bg.ndata['x'].permute(0, 2, 1).to(device)
        images = images.to(device) 
        labels = labels.to(device).squeeze(-1)
        images = images.permute(0,3,1,2)
        
        pred_out, embedding = model(bg, feat, images)
        p_r = dist.Bernoulli(logits=pred_out)
        pred_out = p_r.probs # > 0.2
        #print(p_r.probs , pred_out)
        #raise "err"
        #print(pred_out[0].permute(1,2,0).shape)
        im = Image.fromarray(pred_out[0].permute(1,2,0).detach().cpu().squeeze(-1).numpy()*255.0)
        im = im.convert('RGB') 
        im = ImageOps.mirror(im)
        im.save(img_dir + "/pred_{}.png".format(epoch))
        im = Image.fromarray(images[0].permute(1,2,0).detach().cpu().squeeze(-1).numpy()*255.0)
        im = im.convert('RGB') 
        im = ImageOps.mirror(im)
        im.save(img_dir + "/real_{}.png".format(epoch))
        break
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="", help='Path to saved state')
    parser.add_argument("--no-cuda", action='store_true', help='Do not use CUDA')
    parser.add_argument("--plot_confusion_matrix", action='store_true', help='Plot the confusion matrix')
    parser.add_argument("--seed",  default=0, help='Seed')
    parser.add_argument("--device",  default=0, help='device num')
    parser.add_argument("--times", type=int, default=5)
    parser.add_argument("--img_dir", type=str, default="./tmp", help='Path to save images')
    args = parser.parse_args()
    
    print(args)
    
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
    # Load everything from state
    if len(args.state) == 0:
        raise ValueError("Expected a valid state filename")

    state = torch.load(args.state, map_location='cpu')
    print(state['args'])
   

    device = "cuda:" + str(args.device)
    
    test_dset = FontWiresWithImages(state['args'].dataset_path,  split="test", in_memory=True, shape_type= state['args'].shape_type)
    test_loader = helper.get_dataloader(
        test_dset, state['args'].batch_size, train=True, collate_fn=collate_graphs_with_images)
    
    array_clustering_acc = []
    array_loss = []
    
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    for t in range(args.times):
        
        model = Model(state['args']).to(device)
            
        model.load_state_dict(state['model'])

        print("Running experiment {}/{} times".format(t + 1, args.times))
        cluster_acc = cluster(args, model, test_loader, device)
        array_clustering_acc.append(cluster_acc)
        
        test_loss = val_image_loss(model, test_loader, t, device)
        array_loss.append(test_loss)
        visualize(args, model, test_loader, t, device, args.img_dir)
        
        

    print("Loss accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_loss), np.std(array_loss)))
    print("Clustering accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_clustering_acc), np.std(array_clustering_acc)))
    print("=====================================================")
    
if __name__ == '__main__':
    main()