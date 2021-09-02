import logging
import os
import os.path as osp

import matplotlib
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import helper
from networks import decoder
from networks import face_model
from networks import graph_model
from networks.classifier import LinearClassifier
from solid_mnist import SolidMNISTWithPointclouds, collate_with_pointclouds

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
import parse_util
import chamfer
import random 

chamfer_dist = chamfer.ChamferLoss()

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.nurbs_feat_ext = face_model.get_face_model(
            args.nurbs_model_type, mask_mode=args.mask_mode, area_as_channel=args.area_as_channel, output_dims=args.nurbs_emb_dim)
        self.brep_feat_ext = graph_model.get_graph_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
        self.project_net = nn.Sequential(
                                nn.Linear(args.graph_emb_dim, args.latent_dim),
                                nn.ReLU()
                            )
        self.decoder = decoder.get_decoder_model(args.decoder_type, 1024, args.latent_dim)   

    def forward(self, bg, feat):
        out  = self.nurbs_feat_ext(feat)
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        embedding = self.project_net(graph_emb)
        out = self.decoder(embedding)
        return out, embedding

def val_chamfer_loss(model, loader, epoch, device):
    model.eval()
    total_loss_array = []
    with torch.no_grad():
        for _, (bg, points, labels, _) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device) 
            points = points.to(device) 
            labels = labels.to(device).squeeze(-1)
            pred_out, embedding = model(bg, feat)

            loss = chamfer_dist(points, pred_out) * 1000
            total_loss_array.append(loss.item())
    avg_loss = np.mean(total_loss_array)
    print("[Val] Epoch {:03} Chamfer Loss {:2.3f}".format(epoch, avg_loss.item()))
    print("#########################################")
    return avg_loss     
    
def train_one_epoch(model, loader, optimizer, scheduler, epoch, iteration, device):
    model.train()
    total_loss_array = []
    #iterations = 0
    for _, (bg, points, labels, _) in enumerate(loader):
        iteration = iteration + 1
        optimizer.zero_grad()
    
        feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device) 
        points = points.to(device) 
        labels = labels.to(device).squeeze(-1)
        pred_out, embedding = model(bg, feat)

        # points and points_reconstructed are n_points x 3 matrices
        #dist1, dist2 = chamfer_dist(points, pred_out)
        #loss = (torch.mean(dist1)) + (torch.mean(dist2))
        loss = chamfer_dist(points, pred_out) * 1000

        loss.backward()
        optimizer.step()
        total_loss_array.append(loss.item())
        if iteration % 200 == True:
            avg_loss = np.mean(total_loss_array)
            print("[Train] Epoch {:03}, Iteration {}, Chamfer Loss {:2.3f}".format(epoch, iteration, avg_loss.item()))
            
    scheduler.step()

    avg_loss = np.mean(total_loss_array)
    print("[Train] Epoch {:03} Chamfer Loss {:2.3f}".format(epoch, avg_loss.item()))
    
    return avg_loss


def experiment_name(args, suffix='') -> str:
    """
    Create a name for the experiment from the command line arguments to the script
    :param args: Arguments parsed by argparse
    :param suffix: Suffix string to append to experiment name
    :return: Experiment name as a string
    """
    from datetime import datetime
    tokens = ["AutoPC", args.brep_model_type, args.nurbs_model_type, "mask_" + args.mask_mode, "area_channel_" + str(args.area_as_channel),
               "brep_" + str(args.graph_emb_dim), "nurbs_" + str(args.nurbs_emb_dim), "latent_" + str(args.latent_dim)]
    if args.split_suffix is not None:
        tokens.append(f"split_suffix{args.split_suffix}")
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(suffix) > 0:
        tokens.append(suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = parse_util.get_train_parser("NURBS-Net autoencoder for unstructured pointcloud reconstruction from B-rep solids")

    # B-rep face
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
    parser.add_argument('--decoder_type', type=str,  default='point_decoder',
                        help='Decoder for reconstruction')
    parser.add_argument('--latent_dim', type=int,
                        default=1024, help='Latent vector dimension for encoder')
    parser.add_argument('--shape_type', type=str,
                        default=None, help='upper or lower')
    parser.add_argument("--num_points", type=int, default=1024,
                        help='number of points for pointclouds')
    # Data augmentation
    parser.add_argument('--npy_dataset_path', type=str, default=None, help='Path to pointcloud dataset')
    parser.add_argument('--split_suffix', type=str, default='', help='Suffix for dataset split folders')
    parser.add_argument('--apply_square_symmetry', type=float, default=0.3,
                        help='Probability of applying square symmetry transformation to uv domain')
    
    args, _ = parser.parse_known_args()
    return args


def sketch_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], color='dimgray', edgecolors='k', linewidths=0.5, zdir='x')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.axis("off")
    ax.grid(False)
    return fig


def write_point_cloud_csv(pc, filename):
    np.savetxt(filename, pc, delimiter=",", header="x,y,z")


def visualize(model, loader, epoch, iteration, device, img_dir):
    model.eval()
    bg, points, labels, _ = next(iter(loader))
    feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device) 
    points = points.to(device)
    labels = labels.to(device).squeeze(-1)
    pred_out, embedding = model(bg, feat)
    fig_pred = sketch_point_cloud(pred_out[0].detach().cpu().numpy())
    fig_pred.savefig(os.path.join(img_dir, "pred_{}.png".format(epoch)))
    fig_gt = sketch_point_cloud(points[0].detach().cpu().numpy())
    fig_gt.savefig(os.path.join(img_dir, "gt_{}.png".format(epoch)))
    write_point_cloud_csv(pred_out[0].detach().cpu().numpy(), os.path.join(img_dir, "pred_{}.csv".format(epoch)))
    write_point_cloud_csv(points[0].detach().cpu().numpy(), os.path.join(img_dir, "gt_{}.csv".format(epoch)))


def cluster(model, loader, epoch, iteration, device):
    model.eval()
    embeddings, labels = get_embedding(model, loader, device)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=100)
    kmeans.fit(embeddings)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(np.squeeze(labels), pred_labels)
    logging.info("[Val] NMI score {}".format(score))
    return score


def get_embedding(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for _, (bg, _, label, _) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
            label = label.to(device)#.squeeze(-1)
            pred_out, embedding = model(bg, feat)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels


def classify(autoencoder_model, train_loader, val_loader, device, num_classes):
    temb, tlabels = get_embedding(autoencoder_model, train_loader, device)
    vemb, vlabels = get_embedding(autoencoder_model, val_loader, device)
    emb_dim = temb[0].size(-1)
    clf = LinearClassifier(emb_dim, num_classes).to(device)


    CLASSIFIER_EPOCHS = 100
    CLASSIFIER_BATCH_SIZE = 128

    optimizer = helper.get_optimizer("Adam", clf, lr=0.0001, weight_decay=1e-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS, 0.000001)
    
    emb_train_loader = helper.get_dataloader(list(zip(temb, tlabels)), batch_size=CLASSIFIER_BATCH_SIZE, train=True)
    emb_val_loader = helper.get_dataloader(list(zip(vemb, vlabels)), batch_size=CLASSIFIER_BATCH_SIZE, train=False)

    # Train classifier
    clf.train()
    for iteration in range(CLASSIFIER_EPOCHS):
        true = []
        pred = []
        losses = []
        for (feat, labels) in emb_train_loader:
            optimizer.zero_grad()
            feat, labels = feat.to(device), labels.to(device)
            labels = labels.to(device).squeeze(-1)
            logits = clf(feat)
            #print("logits: ", logits.shape)
            #print("Label size: ", labels.shape)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            true.append(labels.detach().cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        true = np.concatenate(true)
        pred = np.concatenate(pred)
        acc = metrics.accuracy_score(true, pred)
        avg_loss = np.mean(losses)
        if iteration == CLASSIFIER_EPOCHS - 1:
            logging.info("[Train] Classifier Loss {:2.3f}, Acc {}".format(avg_loss, acc))

    # Test classifier
    clf.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for (feat, labels) in emb_val_loader:
            feat, labels = feat.to(device), labels.to(device)
            labels = labels.to(device).squeeze(-1)
            logits = clf(feat)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss_array.append(loss.item())
            true.append(labels.detach().cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
    logging.info("[Val]   Classifier Loss {:2.3f}, Acc {}".format(avg_loss.item(), acc))
    return avg_loss, acc


def main():
    args = parse()
    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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

    train_dset = SolidMNISTWithPointclouds(bin_root_dir=args.dataset_path, npy_root_dir=args.npy_dataset_path, split="train", shape_type=args.shape_type, apply_square_symmetry=args.apply_square_symmetry, split_suffix=args.split_suffix)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, collate_fn=collate_with_pointclouds,
                              pin_memory=True, shuffle=True)
    val_dset = SolidMNISTWithPointclouds(bin_root_dir=args.dataset_path, npy_root_dir=args.npy_dataset_path, split="val", shape_type=args.shape_type, apply_square_symmetry=args.apply_square_symmetry, split_suffix=args.split_suffix)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, collate_fn=collate_with_pointclouds,
                            pin_memory=True, shuffle=True)

    model = Model(args).to(device)

    optimizer = helper.get_optimizer(args.optimizer, model, lr=args.lr, weight_decay=1e-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0.000001)

    iteration  = 0 
    best_loss = float("inf")
    best_acc = 0

    check_every = 5

    for epoch in range(0, args.epochs + 1):
        print("#########################################")
        tloss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, iteration, device)
        if epoch % check_every == 0: 
            #classify(model, train_loader, val_loader, device, train_dset.num_classes)
            #acc = cluster(model, val_loader, epoch, iteration, device)
            visualize(model, val_loader, epoch, iteration, device, img_dir)
            test_loss = val_chamfer_loss(model, val_loader, epoch, device)
            #if best_acc < acc:
                #best_acc = acc
                #helper.save_checkpoint(osp.join(checkpoint_dir, '{}.pt'.format("cluster_best")), model, optimizer, scheduler, args=args)
            helper.save_checkpoint(osp.join(checkpoint_dir, 'reconstruction_last.pt'), model,
                                   optimizer, scheduler, args=args)
            if best_loss > test_loss:
                best_loss = test_loss
                helper.save_checkpoint(osp.join(checkpoint_dir, 'reconstruction_best.pt'), model,
                                       optimizer, scheduler, args=args)
    logging.info("The best clustering accuracy {} and validation loss {}".format(best_acc, best_loss))


if __name__ == '__main__':
    main()
