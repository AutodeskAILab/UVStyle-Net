import os.path as osp
import random

import matplotlib
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cluster(embeddings, labels):
    kmeans = KMeans(init="k-means++", n_clusters=26, n_init=100)
    kmeans.fit(embeddings)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(np.squeeze(labels), pred_labels)
    print("NMI score {}".format(score))
    return score


def get_embedding_solid(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for _, (bg, _, label) in enumerate(loader):
            feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
            label = label.to(device)  # .squeeze(-1)
            # points = points.to(device)
            pred_out, embedding = model(bg, feat)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels


def get_embedding_pc(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for _, (_, points, label) in enumerate(loader):
            label = label.to(device)  # .squeeze(-1)
            points = points.to(device)
            pred_out, embedding = model(points.transpose(-1, 1))
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels


def visualize_pc(batch_pc_pred, batch_pc_gt, filename, img_dir):
    multiple_plot_label_pc(
        batch_pc_gt.detach().cpu().numpy(),
        batch_pc_pred.detach().cpu().numpy(),
        4,
        filenames=[f"gt_vs_pred_{fn}" for fn in filename],
        save_loc=img_dir,
    )

    np.savetxt(
        osp.join(img_dir, f"pred_{filename[0]}.csv"),
        batch_pc_pred[0].detach().cpu().numpy(),
        delimiter=",",
        header="x,y,z",
    )
    np.savetxt(
        osp.join(img_dir, f"gt_{filename[0]}.csv"),
        batch_pc_gt[0].detach().cpu().numpy(),
        delimiter=",",
        header="x,y,z",
    )


def multiple_plot_label_pc(batch_real, batch_pred, num_plots, filenames, save_loc=None):

    for i in range(num_plots):
        fig = plt.figure()

        ax = fig.add_subplot("121", projection="3d")
        real = batch_real[i]

        ax.scatter(
            real[:, 0],
            real[:, 1],
            real[:, 2],
            color="dimgray",
            edgecolors="k",
            linewidths=0.5,
        )
        ax.view_init(-225, 90)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.axis("off")
        ax.grid(False)

        ax = fig.add_subplot("122", projection="3d")
        pred = batch_pred[i]

        ax.scatter(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            color="dimgray",
            edgecolors="k",
            linewidths=0.5,
        )
        ax.view_init(-225, 90)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.axis("off")
        ax.grid(False)

        save_img_loc = osp.join(save_loc, filenames[i] + ".png")
        if save_loc != None:
            plt.savefig(save_img_loc)
            plt.close()
        else:
            plt.show()

    return


def retrieve_nearest(
    embeddings, loader, num_retrievals=1, num_retrievals_per_query=5, batch_size=32
):
    retrievals = {}
    for _ in range(num_retrievals):
        query_object_ind = random.randint(0, len(embeddings))
        query_object_name = loader.dataset.pc_files[query_object_ind]
        query_embedding = embeddings[query_object_ind]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        dist_matrix = cdist(query_embedding, embeddings, metric="euclidean")
        idx = dist_matrix[0].argsort()[1 : num_retrievals_per_query + 1]
        nearest = []
        for i in range(idx.size):
            nearest.append(loader.dataset.pc_files[idx[i]])
        print(f"Query: {query_object_name}")
        for item in nearest:
            print(f"Nearest: {item}")
        retrievals[query_object_name] = nearest
        print("--------------------------------------------")
    return retrievals
