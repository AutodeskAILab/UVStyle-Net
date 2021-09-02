import torch.utils.tensorboard as tb

import pandas as pd
import chamfer
import random
import parse_util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import helper
import os.path as osp
from torch.optim import lr_scheduler
import sklearn.metrics as metrics
import scipy
import scipy.spatial
import numpy as np
import os
from networks import nurbs_model
from networks import brep_model
from networks import decoder, pointnet
from networks.classifier import LinearClassifier
from abcdataset import ABCDatasetWithPointclouds
import matplotlib
import h5py

from test_classifier import log_activation_stats

matplotlib.use("Agg")
chamfer_dist = chamfer.ChamferLoss()


def plot_tsne(data_points, labels=None, save_loc=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=5000)
    tsne_results = tsne.fit_transform(data_points)
    my_colors = {0: 'orange', 1: 'red', 2: 'green', 3: 'blue', 4: 'grey', 5: 'gold', 6: 'violet', 7: 'pink', 8: 'navy',
                 9: 'black'}
    if labels != None:
        for i, data_point in enumerate(tsne_results):
            plt.scatter(data_point[0], data_point[1], color=my_colors.get(labels[i], 'black'))
    else:
        for i, data_point in enumerate(tsne_results):
            plt.scatter(data_point[0], data_point[1], color='orange')
    if save_loc != None:
        plt.savefig(save_loc)
        plt.close()
        return
    plt.show()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.nurbs_feat_ext = nurbs_model.get_face_model(
            args.nurbs_model_type,
            mask_mode=args.mask_mode,
            area_as_channel=args.area_as_channel,
            output_dims=args.nurbs_emb_dim,
        )
        self.brep_feat_ext = brep_model.get_graph_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim
        )
        self.project_net = nn.Sequential(
            nn.Linear(args.graph_emb_dim, args.latent_dim), nn.ReLU()
        )
        self.decoder = decoder.get_decoder_model(
            args.decoder_type, 1024, args.latent_dim
        )
        self.th = nn.Tanh()

    def forward(self, bg, feat):
        out = self.nurbs_feat_ext(feat)
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        embedding = self.project_net(graph_emb)
        out = self.decoder(embedding)
        out = self.th(out)
        return out, embedding


def val_chamfer_loss(model, loader, device, img_dir):
    model.eval()
    total_loss_array = []
    with torch.no_grad():
        for idx, (bg, points, _) in enumerate(loader):
            try:
                feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
                points = points.to(device)
            except:
                continue
            pred_out, embedding = model(bg, feat)
            np.savetxt(
                osp.join(img_dir, loader.dataset.pc_files[idx] + ".csv"),
                pred_out[0].detach().cpu().numpy(),
                delimiter=",",
                header="x,y,z",
            )
            loss = chamfer_dist(points, pred_out) * 1000
            total_loss_array.append(loss.item())
    avg_loss = np.mean(total_loss_array)
    print("[Test] Chamfer Loss {:2.3f}".format(avg_loss.item()))
    print("#########################################")
    return avg_loss


def get_embedding(model, loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, (bg, _, _) in enumerate(loader):
            if i > 0 and i % 400 == 0 or i == len(loader) - 1:
                print(f"Computing embeddings: {i}/{len(loader)} ...")
            try:
                feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
            except:
                continue
            pred_out, embedding = model(bg, feat)
            embeddings.append(embedding.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    return embeddings


def save_all_embeddings(model, loader, device):
    model.eval()
    all_graph_files = []
    stats = {}
    with torch.no_grad():
        for i, (bg, _, graph_files) in enumerate(loader):
            print('progress:', i / len(loader))
            try:
                feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
            except:
                continue
            pred_out, embedding = model(bg, feat)
            for activations in [model.nurbs_feat_ext.activations, model.brep_feat_ext.activations]:
                batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]
            all_graph_files += graph_files

    print('writing stats...')
    all_stats = {}
    for layer, layer_stats in stats.items():
        mean, sigma, cov, gram = zip(*layer_stats)
        all_stats[layer] = {
            'mean': torch.cat(mean),
            'sigma': torch.cat(sigma),
            'cov': torch.cat(cov),
            'gram': torch.cat(gram),
        }

    for i, (layer, layer_stats) in enumerate(all_stats.items()):
        grams = layer_stats['gram'].numpy()
        np.save(f'{out_dir}/{i}_{layer}_grams', grams)
    all_graph_files = list(map(lambda file: file.split('/')[-1], all_graph_files))
    pd.DataFrame(all_graph_files).to_csv(out_dir + '/graph_files.txt', index=False, header=None)
    print('done writing stats')



def embedding_tsne(model, loader, device):
    embeddings = get_embedding(model, loader, device)
    plot_tsne(embeddings, labels=None, save_loc="abc_tsne")


def retrieve_nearest(model, loader, device, num_retrievals=1):
    print("Retrieval based on nearest neighbors in latent space")
    embeddings = get_embedding(model, loader, device)
    for _ in range(num_retrievals):
        query_object_ind = random.randint(0, len(embeddings))
        query_object_name = loader.dataset.graph_files[query_object_ind]
        query_embedding = embeddings[query_object_ind]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        dist_matrix = scipy.spatial.distance.cdist(
            query_embedding, embeddings, metric="euclidean"
        )
        idx = dist_matrix[0].argsort()[1:6]  # top 5
        nearest = []
        for i in range(idx.size):
            nearest.append(loader.dataset.graph_files[idx[i]])
        print(f"Query: {query_object_name}")
        for item in nearest:
            print(f"Nearest: {item}")
        print("--------------------------------------------")
    # util.sketch_nearest(query_object, nearest_pcs)
    return


def main():
    out_dir = 'analysis/uvnet_data/abc'
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="", help="Path to saved state")
    parser.add_argument("--binary_dir", type=str, help="Path to dataset binary files")
    parser.add_argument("--pc_dir", type=str, help="Path to dataset point cloud files")
    parser.add_argument("--no-cuda", action="store_true", help="Do not use CUDA")
    parser.add_argument("--device", default=0, help="device num")
    parser.add_argument("--seed", default=0, help="Random seed")
    parser.add_argument("--img_dir", type=str, default="")
    args = parser.parse_args()
    print(args)
    # Load everything from state
    if len(args.state) == 0:
        raise ValueError("Expected a valid state filename")
    state = helper.load_checkpoint(args.state, map_to_cpu=args.no_cuda)
    print("Args used during training:\n", state["args"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.no_cuda:
        device = 'cpu'
    else:
        device = "cuda:" + str(args.device)
    # args.device = device
    test_dset = ABCDatasetWithPointclouds(
        args.binary_dir,
        args.pc_dir,
        split="test",
        apply_square_symmetry=0.0,
    )
    entire_dset = ABCDatasetWithPointclouds(
        state["args"].dataset_path,
        state["args"].npy_dataset_path,
        split="all",
        apply_square_symmetry=0.0,
    )
    test_loader = test_dset.get_dataloader(32, shuffle=False)
    entire_loader = test_loader
    # entire_loader = entire_dset.get_dataloader(1, shuffle=False)
    # Test chamfer loss and save pointcloud reconstructions
    array_chamfer_loss = []
    model = Model(state["args"]).to(device)
    model.load_state_dict(state["model"])
    save_all_embeddings(model, entire_loader, device)
    test_loss = val_chamfer_loss(model, test_loader, device, args.img_dir)
    print(f"Chamfer loss: {test_loss}")

    embedding_tsne(model, entire_loader, device)
    # Retrieval on entire set
    # retrieve_nearest(model, entire_loader, device, num_retrievals=10)


if __name__ == "__main__":
    writer = tb.SummaryWriter()
    main()
