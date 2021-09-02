import argparse
import os

import matplotlib
# from parse import Parser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import helper
from networks import brep_model
from networks import decoder, pointnet
from networks import nurbs_model
from solid_mnist import SolidMNISTWithPointclouds, collate_with_pointclouds
from test_classifier import log_activation_stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
import random

import chamfer

chamfer_dist = chamfer.ChamferLoss()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.nurbs_feat_ext = nurbs_model.get_face_model(
            args.nurbs_model_type, mask_mode=args.mask_mode, area_as_channel=args.area_as_channel,
            output_dims=args.nurbs_emb_dim)
        self.brep_feat_ext = brep_model.get_graph_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
        self.project_net = nn.Sequential(
            nn.Linear(args.graph_emb_dim, args.latent_dim),
            nn.ReLU()
        )
        self.decoder = decoder.get_decoder_model(args.decoder_type, 1024, args.latent_dim)

    def forward(self, bg, feat):
        out = self.nurbs_feat_ext(feat)
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        embedding = self.project_net(graph_emb)
        out = self.decoder(embedding)
        return out, embedding


class Model_Pointnet(nn.Module):
    def __init__(self, args):
        super(Model_Pointnet, self).__init__()

        self.encoder = pointnet.PointNet_Encoder(args.latent_dim)
        self.decoder = decoder.get_decoder_model(args.decoder_type, 1024, args.latent_dim)

    def forward(self, pc):
        embedding = self.encoder(pc)
        out = self.decoder(embedding)
        return out, embedding


def val_chamfer_loss(args, model, loader, epoch, device):
    model.eval()
    total_loss_array = []
    with torch.no_grad():
        for _, (bg, points, labels) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
            points = points.to(device)
            labels = labels.to(device).squeeze(-1)
            if args.encoder_type == "pointnet":
                pred_out, embedding = model(points.transpose(-1, 1))
            else:
                pred_out, embedding = model(bg, feat)

            loss = chamfer_dist(points, pred_out) * 1000
            total_loss_array.append(loss.item())
    avg_loss = np.mean(total_loss_array)
    print("[Val] Epoch {:03} Chamfer Loss {:2.3f}".format(epoch, avg_loss.item()))
    print("#########################################")
    return avg_loss


def multiple_plot_label_pc(batch_real, batch_pred, num_plots, save_loc=None):
    for i in range(num_plots):
        fig = plt.figure()

        ax = fig.add_subplot("121", projection='3d')
        real = batch_real[i]

        ax.scatter(real[:, 0], real[:, 1], real[:, 2], color='dimgray', edgecolors='k', linewidths=0.5, )
        ax.view_init(-225, 90)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.axis("off")
        ax.grid(False)

        ax = fig.add_subplot("122", projection='3d')
        pred = batch_pred[i]

        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='dimgray', edgecolors='k', linewidths=0.5, )
        ax.view_init(-225, 90)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.axis("off")
        ax.grid(False)

        save_img_loc = save_loc + str(i) + ".png"
        if save_loc != None:
            plt.savefig(save_img_loc)
            plt.close()
        else:
            plt.show()

    return


def sketch_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='dimgray', edgecolors='k', linewidths=0.5, zdir='x')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.axis("off")
    ax.grid(False)
    return fig


def write_point_cloud_csv(pc, filename):
    np.savetxt(filename, pc, delimiter=",", header="x,y,z")


def visualize(args, model, loader, epoch, device, img_dir):
    model.eval()
    bg, points, labels, _ = next(iter(loader))
    feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
    points = points.to(device)
    labels = labels.to(device).squeeze(-1)
    if args.encoder_type == "pointnet":
        pred_out, embedding = model(points.transpose(-1, 1))
    else:
        pred_out, embedding = model(bg, feat)
    # pred_out, embedding = model(bg, feat)
    # print(points.shape, pred_out.shape)
    multiple_plot_label_pc(points.detach().cpu().numpy(), pred_out.detach().cpu().numpy(), 4,
                           save_loc=img_dir + "{}_".format(epoch))

    # fig_pred = sketch_point_cloud(pred_out[0].detach().cpu().numpy())
    # fig_pred.savefig(os.path.join(img_dir, "pred_{}.png".format(epoch)))
    # fig_gt = sketch_point_cloud(points[0].detach().cpu().numpy())
    # fig_gt.savefig(os.path.join(img_dir, "gt_{}.png".format(epoch)))
    write_point_cloud_csv(pred_out[0].detach().cpu().numpy(), os.path.join(img_dir, "pred_{}.csv".format(epoch)))
    write_point_cloud_csv(points[0].detach().cpu().numpy(), os.path.join(img_dir, "gt_{}.csv".format(epoch)))


def cluster(args, model, loader, device):
    model.eval()
    embeddings, labels = get_embedding(args, model, loader, device)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=100)
    kmeans.fit(embeddings)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(np.squeeze(labels), pred_labels)
    print("[Val] NMI score {}".format(score))
    return score


def get_embedding(args, model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    stats = {}
    all_graph_files = []
    with torch.no_grad():
        for _, (bg, points, label, graph_files) in enumerate(loader):
            feat = bg.ndata['x'].permute(0, 3, 1, 2).to(device)
            label = label.to(device)  # .squeeze(-1)
            points = points.to(device)
            if args.encoder_type == "pointnet":
                pred_out, embedding = model(points.transpose(-1, 1))
            else:
                pred_out, embedding = model(bg, feat)
                for activations in [model.nurbs_feat_ext.activations, model.brep_feat_ext.activations]:
                    batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]
                all_graph_files += graph_files
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    print('writing stats...')
    all_stats = {}
    for layer, layer_stats in stats.items():
        mean, sigma, cov, gram = zip(*layer_stats)
        all_stats[layer] = {
            'gram': torch.cat(gram),
        }

    for i, (layer, layer_stats) in enumerate(all_stats.items()):
        grams = layer_stats['gram'].numpy()
        np.save(f'{i}_{layer}_grams', grams)

    all_graph_files = list(map(lambda file: file.split('/')[-1], all_graph_files))
    pd.DataFrame(all_graph_files).to_csv('analysis/uvnet_data/solidmnist_font_subset_pc_reconstruct/graph_files.txt', index=False, header=None)
    print('done writing stats')
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, help='Path to saved state',
                        default='/Users/t_meltp/OneDrive/models/uvnet/pc_reconstruct/reconstruction_best.pt')
    parser.add_argument("--encoder_type", type=str, default="brep", help='brep or pointnet')
    parser.add_argument("--no-cuda", action='store_true', help='Do not use CUDA')
    parser.add_argument("--plot_confusion_matrix", action='store_true', help='Plot the confusion matrix')
    parser.add_argument("--seed", default=0, help='Seed')
    parser.add_argument("--device", default=0, help='device num')
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--img_dir", type=str, default="./tmp", help='Path to save images')
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

    if args.no_cuda:
        device = 'cpu'
    else:
        device = "cuda:" + str(args.device)
    # args.device = device

    test_dset = SolidMNISTWithPointclouds(bin_root_dir='/Users/t_meltp/solid-mnist/bin',
                                          npy_root_dir='/Users/t_meltp/solid-mnist/pc', split="test",
                                          num_points=state['args'].num_points, shape_type=state['args'].shape_type,
                                          split_suffix=state['args'].split_suffix)

    test_loader = helper.get_dataloader(
        test_dset, 32, train=False, collate_fn=collate_with_pointclouds)

    # Train/validate

    array_clustering_acc = []
    array_chamfer_loss = []

    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    for t in range(args.times):

        if args.encoder_type == "pointnet":
            model = Model_Pointnet(state['args']).to(device)
        else:
            model = Model(state['args']).to(device)

        model.load_state_dict(state['model'])

        visualize(args, model, test_loader, t, device, args.img_dir)

        print("Running experiment {}/{} times".format(t + 1, args.times))
        cluster_acc = cluster(args, model, test_loader, device)
        array_clustering_acc.append(cluster_acc)

        test_loss = val_chamfer_loss(args, model, test_loader, t, device)
        array_chamfer_loss.append(test_loss)

    print("Chamfer accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_chamfer_loss), np.std(array_chamfer_loss)))
    print("Clustering accuracy: {:2.7f}+-{:2.7f}".format(np.mean(array_clustering_acc), np.std(array_clustering_acc)))
    print("=====================================================")


if __name__ == '__main__':
    main()
