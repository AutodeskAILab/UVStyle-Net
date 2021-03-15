import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.neighbors._kd_tree import KDTree

from util import Grams, weight_layers, Images, IdMap, OnTheFlyImages


def spherize(emb):
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / norm


if __name__ == '__main__':
    uv_net_data_root = '../../uvnet_data/solidmnist_sub_mu_only'
    pointnet_data_root = '../../pointnet2_data/solidmnist_font_subset'
    meshcnn_data_root = '../../meshcnn_data/solidmnist_font_subset'
    psnet_data_root = '../../psnet_data/solidmnist_subset'
    img_path = '/home/pete/brep_style/solidmnist/new_pngs'

    grams = {
        'UV-Net': Grams(data_root=uv_net_data_root),
        'PSNet*': Grams(data_root=psnet_data_root),
        'PointNet++': Grams(data_root=pointnet_data_root),
        'MeshCNN': Grams(data_root=meshcnn_data_root)
    }

    images = {
        'UV-Net': OnTheFlyImages(data_root=uv_net_data_root,
                                 img_root=img_path),
        'PointNet++': OnTheFlyImages(data_root=pointnet_data_root,
                                     img_root=img_path),
        'PSNet*': OnTheFlyImages(data_root=psnet_data_root,
                                   img_root=img_path),
        'MeshCNN': OnTheFlyImages(data_root=meshcnn_data_root,
                                  img_root=img_path)
    }

    layers = {
        'UV-Net': [[0], [2], [3]],
        'PSNet*': [[0], [2], [3]],
        'PointNet++': [[0], [5], [10]],
        'MeshCNN': [[0], [1], [2]]
    }

    layer_names = {
        'UV-Net': ['0_feats', '2_conv2', '3_conv3'],
        'PSNet*': ['0_feats', '2_conv2', '3_conv3'],
        'PointNet++': ['0_feats', '5_SA1_2_2', '10_SA2_1_1'],
        'MeshCNN': ['0_feats', '1_conv1', '2_conv2']
    }

    fig, axes = plt.subplots(len(grams), squeeze=True)
    for i, (model, gram) in enumerate(grams.items()):
        query_idx = 53
        if model == 'PointNet++':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=pointnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'MeshCNN':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=meshcnn_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'PSNet*':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=psnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)

        results = []
        errors = []
        for layer_selection in layers[model]:
            weights = np.zeros(len(gram))
            weights[layer_selection] = 1.

            combined = weight_layers(gram.grams, weights)
            combined = spherize(combined)
            kd_tree = KDTree(combined)
            queries = [combined[query_idx]]
            imgs = kd_tree.query(queries, k=6, return_distance=False)
            print(imgs)
            print(gram.labels[imgs])
            results.append(imgs)
            labels = gram.labels[imgs]
            error = labels != labels[0, 0]
            errors.append(error)
        results = np.concatenate(results, axis=-1).flatten()
        imgs_to_show = list(map(lambda r: images[model][r][0], results))
        imgs_to_show = map(torchvision.transforms.Resize((256, 256)), imgs_to_show)
        img_tensors = map(torchvision.transforms.ToTensor(), imgs_to_show)


        def highlight_err(img, err):
            if err:
                thickness = 10
                color = torch.tensor([1., 0., 0.])
                x = img.permute(1, 2, 0)
                x[:thickness, :, :] = color
                x[-thickness:, :, :] = color
                x[:, :thickness, :] = color
                x[:, -thickness:, :] = color
                return x.permute(2, 0, 1)
            else:
                return img


        img_tensors = [highlight_err(img, err) for img, err in zip(img_tensors, np.array(errors).flatten())]
        grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))
        ax = axes[i]
        ax.imshow(grid)

        ax.set_ylabel(model, Fontsize=20)
        ax.set_yticks(np.arange(3) * 256 + 128)
        ax.set_yticklabels(layer_names[model], fontsize='small', rotation=60)

        ax.set_xticks(np.arange(6) * 256 + 128)
        ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'], Fontsize=20)
    fig.set_size_inches(5, 10)
    fig.tight_layout()
    fig.savefig(f'low_mid_high.pdf')
    plt.show()
