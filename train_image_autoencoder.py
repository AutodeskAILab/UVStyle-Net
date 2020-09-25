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
from PIL import Image


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

def visualize(model, loader, epoch, iteration, device, img_dir):
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
        im.save(img_dir + "/pred_{}.png".format(epoch))
        im = Image.fromarray(images[0].permute(1,2,0).detach().cpu().squeeze(-1).numpy()*255.0)
        im = im.convert('RGB') 
        im.save(img_dir + "/real_{}.png".format(epoch))
        break
    
        
        
def train_one_epoch(model, loader, optimizer, scheduler, epoch, iteration, device):
    model.train()
    total_loss_array = []
    for _, (bg, images, labels) in enumerate(loader):
        iteration = iteration + 1
        optimizer.zero_grad()
    
        feat = bg.ndata['x'].permute(0, 2, 1).to(device)
        images = images.to(device) 
        labels = labels.to(device).squeeze(-1)
        images = images.permute(0,3,1,2)

        pred_out, embedding = model(bg, feat, images)
        
        #loss = F.binary_cross_entropy_with_logits(
            #pred_out, images)#.sum((1,2,3)).mean() #.mean()
        
        p_r = dist.Bernoulli(logits=pred_out)
        logits  = p_r.logits
        loss = F.binary_cross_entropy_with_logits(
            logits, images)#.sum((1,2,3)).mean() #.mean()
        loss.backward()
        optimizer.step()
        total_loss_array.append(loss.item())
        if iteration % 200 == True:
            avg_loss = np.mean(total_loss_array)
            print("[Train] Epoch {:03}, Iteration {}, Loss {:2.3f}".format(epoch, iteration, avg_loss.item()))
            
    scheduler.step()

    avg_loss = np.mean(total_loss_array)
    print("[Train] Epoch {:03}  Loss {:2.3f}".format(epoch, avg_loss.item()))
    
    return avg_loss


def experiment_name(args, suffix='') -> str:
   
    from datetime import datetime
    if args.encoder_type == "brep":
        
        tokens = ["AutoImage", args.brep_model_type, args.nurbs_model_type, "mask_" + args.mask_mode, "area_channel_" + str(args.area_as_channel), "brep_" + str(args.graph_emb_dim), "nurbs_" + str(args.nurbs_emb_dim), "latent_" + str(args.latent_dim)]
    else:
        tokens = ["AutoImage", "latent_" + str(args.latent_dim)]
     
    if args.split_suffix is not None:
        tokens.append(f"split_suffix{args.split_suffix}")
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(suffix) > 0:
        tokens.append(suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = parse_util.get_train_parser("curve based autoencoder")

    # B-rep face
    parser.add_argument('--encoder_type', type=str, default='brep', help='image or brep')
    
    parser.add_argument('--nurbs_model_type', type=str, choices=('cnn','wcnn'), default='cnn',
                        help='Feature extractor for NURBS surfaces')
    parser.add_argument('--nurbs_emb_dim', type=int, default=64,
                        help='Embedding dimension for NURBS feature extractor (default: 64)')
    parser.add_argument("--mask_mode", type=str, default="channel", choices=("channel", "multiply"),
                        help="Whether to consider trimming mask as channel or multiply it with computed features")
    parser.add_argument("--area_as_channel", action="store_true",
                        help="Whether to use area as a channel in the input")
    # B-rep graph
    parser.add_argument('--brep_model_type',  type=str, default='gin_grouping',
                        help='Feature extractor for B-rep face-adj graph')
    parser.add_argument('--graph_emb_dim', type=int,
                        default=128, help='Embeddings before graph pooling')
    # Autoencoder

    parser.add_argument('--latent_dim', type=int,
                        default=64, help='Latent vector dimension for encoder')
    parser.add_argument('--shape_type', type=str,
                        default="upper", help='upper or lower')
    parser.add_argument("--num_points", type=int, default=1024,
                        help='number of points for pointclouds')
    # Data augmentation
    parser.add_argument('--size_percentage', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--npy_dataset_path', type=str, default=None, help='Path to pointcloud dataset')
    parser.add_argument('--split_suffix', type=str, default=None, help='Suffix for dataset split folders')
    parser.add_argument('--apply_line_symmetry', type=float, default=0.3,
                        help='Probability of randomly applying a line symmetry transform on the curve grid')
    
    args = parser.parse_args()
    return args

def main():
    args = parse()
    print(args)
    
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
    device = "cuda:" + str(args.device)

    exp_name = experiment_name(args)

    # Create directories for checkpoints and logging
    log_filename = osp.join('dump', exp_name, 'log.txt')
    checkpoint_dir = osp.join('dump', exp_name, 'checkpoints')
    img_dir = osp.join('dump', exp_name, 'imgs')
    helper.create_dir(checkpoint_dir)
    helper.create_dir(img_dir)
    # Setup logger
    helper.setup_logging(log_filename)
    logging.info("Experiment name: {}".format(exp_name))

    train_dset = FontWiresWithImages(args.dataset_path,  split="train", size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry, in_memory=True, shape_type= args.shape_type)
    val_dset = FontWiresWithImages(args.dataset_path, split="val", size_percentage=args.size_percentage, apply_line_symmetry=args.apply_line_symmetry, in_memory=True, shape_type= args.shape_type)
    train_loader = helper.get_dataloader(
        train_dset, args.batch_size, train=True, collate_fn=collate_graphs_with_images)
    val_loader = helper.get_dataloader(
        val_dset, args.batch_size, train=False, collate_fn=collate_graphs_with_images)
    
   
    check_every = 5
    
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
        model = Model(args).to(device)
        logging.info("Model has {} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        optimizer = helper.get_optimizer(args.optimizer, model, lr=args.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0.000001)

        best_acc = 0.0
        logging.info("Running experiment {}/{} times".format(t + 1, args.times))
        
        for epoch in range(0, args.epochs + 1):
            print("#########################################")
            visualize(model, val_loader, epoch, iteration, device, img_dir)
            tloss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, iteration, device)
            #raise "err"

            if epoch % check_every == 0: 
                test_loss = val_image_loss(model, val_loader, epoch, device)
                helper.save_checkpoint(osp.join(checkpoint_dir, 'reconstruction_last.pt'), model,
                                       optimizer, scheduler, args=args)
                if best_loss > test_loss:
                    best_loss = test_loss
                    helper.save_checkpoint(osp.join(checkpoint_dir, 'reconstruction_best.pt'), model,
                                           optimizer, scheduler, args=args)
        val_losses[t, epoch] = best_loss
        
    #logging.info("The best clustering accuracy {} and validation loss {}".format(best_acc, best_loss))


if __name__ == '__main__':
    main()