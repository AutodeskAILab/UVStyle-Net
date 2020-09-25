import argparse
import os.path as osp
import logging
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint, uniform
import scipy
import random 



def plot_tsne(data_points, labels=None, save_loc=None):

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_points)
    my_colors = {0:'orange',1:'red',2:'green',3:'blue',4:'grey',5:'gold',6:'violet',7:'pink',8:'navy',9:'black'}

    if labels.any() == False:
        plt.plot(tsne_results[:,0], tsne_results[:,1],'ro' )
    else:
        for i, data_point in enumerate(tsne_results):
            plt.scatter(data_point[0] , data_point[1], color = my_colors.get(labels[i], 'black'))
            
    if save_loc != None:
        plt.savefig(save_loc)
        return 
    plt.show()

def clustering(n_clusters, data_points, labels):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
    kmeans.fit(data_points)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(labels, pred_labels)
    print("NMI score on training data {}".format(score))
    return score

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def sketch_point_cloud(points, save_loc=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], color='dimgray', edgecolors='k', linewidths=0.5, zdir='-x')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    #ax.axis("off")
    #ax.grid(False)
    if save_loc != None:
        plt.savefig(save_loc)
        return 
    plt.show()


def sketch_point_cloud_based_point_clouds(points, save_loc=None):
    xyz, normals, mask = points[:, 0:3], points[:, 3:6], points[:, 6]
    print(mask, np.nonzero(mask))
    xyz = xyz[np.nonzero(mask)]
    xyz = pc_normalize(xyz)
    normals= normals[np.nonzero(mask)]
    print(xyz.shape, normals.shape, mask.shape)
    sketch_point_cloud(xyz)
    #sketch_point_cloud(normals)


def sketch_nearest(query_object, nearest_objects, save_loc=None):
    fig = plt.figure(figsize=(15,3))
    count = 1
    num_objects = len(nearest_objects)

    ax = fig.add_subplot(1, num_objects + 1,  1, projection='3d')
    ax.scatter(query_object[:,0], query_object[:,1], query_object[:,2], color='dimgray', edgecolors='k', linewidths=0.5, zdir='-x')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.axis("off")
    ax.grid(False)

    for i, pc in enumerate(nearest_objects):
        ax = fig.add_subplot(1, num_objects + 1, count + 1, projection='3d')
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], color='dimgray', edgecolors='k', linewidths=0.5, zdir='-x')
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-0.8, 0.8])
        ax.axis("off")
        ax.grid(False)
        count = count + 1
    fig.tight_layout()

    if save_loc != None:
        plt.savefig(save_loc)
        return 
    plt.show()


def retrieve_neighbours(pcs, embeddings, num_neighbours=5, labels=None, random_index=None,  save_loc=None):
    if random_index == None:
        random_index = random.randint(0, len(embeddings))

    query_object, query_embedding = pcs[random_index], embeddings[random_index]

    if labels != None:
        query_label  = labels[random_index]

    query_embedding = np.expand_dims(query_embedding, axis=0)
    dist_matrix = scipy.spatial.distance.cdist(query_embedding, embeddings, metric='euclidean')
    idx = dist_matrix[0].argsort()[1:num_neighbours]
    nearest_pcs = pcs[idx]
    if labels != None:
        nearest_labels = labels[idx]

    sketch_nearest(query_object, nearest_pcs, save_loc=save_loc)
    return