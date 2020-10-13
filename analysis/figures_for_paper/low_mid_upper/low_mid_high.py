import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.neighbors._kd_tree import KDTree

from util import Grams, weight_layers, Images, IdMap, OnTheFlyImages


def spherize(emb):
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / norm


if __name__ == '__main__':
    uv_net_data_root = '../../uvnet_data/solidmnist_font_subset'
    pointnet_data_root = '../../pointnet2_data/solidmnist_font_subset'
    meshcnn_data_root = '../../meshcnn_data/solidmnist_font_subset'

    grams = {
        'UV-Net': Grams(data_root=uv_net_data_root),
        'Pointnet++': Grams(data_root=pointnet_data_root),
        'MeshCNN': Grams(data_root=meshcnn_data_root)
    }

    images = {
        'UV-Net': OnTheFlyImages(data_root=uv_net_data_root,
                         img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs'),
        'Pointnet++': OnTheFlyImages(data_root=pointnet_data_root,
                             img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs'),
        'MeshCNN': OnTheFlyImages(data_root=meshcnn_data_root,
                                  img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs')
    }

    layers = {
        'UV-Net': [[1], [4], [6]],
        'Pointnet++': [[1], [12], [21]],
        'MeshCNN': [[1], [2], [4]]
    }

    id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                   dest_file=pointnet_data_root + '/graph_files.txt')

    fig, axes = plt.subplots(3, squeeze=True)
    for i, (model, gram) in enumerate(grams.items()):
        query_idx = 121
        if model == 'Pointnet++':
            query_idx = id_map(query_idx)
        elif model == 'MeshCNN':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=meshcnn_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)

        results = []
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
        results = np.concatenate(results, axis=-1).flatten()
        imgs_to_show = list(map(lambda r: images[model][r][0], results))
        imgs_to_show = map(torchvision.transforms.Resize((256, 256)), imgs_to_show)
        img_tensors = list(map(torchvision.transforms.ToTensor(), imgs_to_show))
        grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))
        ax = axes[i]
        ax.imshow(grid)
        ax.set_title(model, size='large')

        ax.set_ylabel('Layer')
        ax.set_yticks(np.arange(3) * 256 + 128)
        ax.set_yticklabels(layers[model])

        ax.set_xticks(np.arange(6) * 256 + 128)
        ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'])
    fig.set_size_inches(4, 6)
    fig.tight_layout()
    fig.savefig(f'low_mid_high.pdf')
    plt.show()