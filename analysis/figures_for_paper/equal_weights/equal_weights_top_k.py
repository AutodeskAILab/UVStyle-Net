import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from joblib import Parallel, delayed
from tqdm import tqdm

from abc_all.abc_top_k_by_layer import gram_loss
from util import Grams, IdMap, OnTheFlyImages

if __name__ == '__main__':
    uv_net_data_root = '../../uvnet_data/solidmnist_sub_mu_only'
    pointnet_data_root = '../../pointnet2_data/solidmnist_font_subset'
    meshcnn_data_root = '../../meshcnn_data/solidmnist_font_subset'
    psnet_data_root = '../../psnet_data/solidmnist_subset'
    img_path = '/home/pete/brep_style/solidmnist/new_pngs'
    grams = {
        'UV-Net': Grams(data_root=uv_net_data_root),
        'Pointnet++': Grams(data_root=pointnet_data_root),
        'MeshCNN': Grams(data_root=meshcnn_data_root),
        'Pointnet': Grams(data_root=psnet_data_root)
    }

    images = {
        'UV-Net': OnTheFlyImages(data_root=uv_net_data_root,
                                 img_root=img_path),
        'Pointnet++': OnTheFlyImages(data_root=pointnet_data_root,
                                     img_root=img_path),
        'MeshCNN': OnTheFlyImages(data_root=meshcnn_data_root,
                                  img_root=img_path),
        'Pointnet': OnTheFlyImages(data_root=psnet_data_root,
                                   img_root=img_path)
    }

    letter_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams['UV-Net'].graph_files))
    }

    fig, axes = plt.subplots(len(grams), squeeze=True)
    for i, (model, gram) in enumerate(grams.items()):
        query_idx = np.array([
            letter_idx['y_Abhaya Libre_upper'],
            letter_idx['n_Zhi Mang Xing_lower'],
            letter_idx['l_Seaweed Script_upper'],
            letter_idx['e_Turret Road_upper']
        ])
        if model == 'Pointnet++':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=pointnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'MeshCNN':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=meshcnn_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'Pointnet':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=psnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)

        # select lower half of layers
        n = len(gram) // 2
        weights = np.zeros(len(gram))
        weights[:n] = 1.

        results = []
        all_distances = []
        num = len(gram.graph_files)
        for query in query_idx:
            distances = np.zeros(num)
            inputs = tqdm(range(num))
            x = Parallel(-1)(delayed(gram_loss)(gram, query, other, weights, metric='cosine') for other in inputs)
            for idx, distance in x:
                distances[idx] = distance
            results.append(np.argsort(distances)[:6])
            all_distances.append(distances[np.argsort(distances)[:6]])
        all_results = np.concatenate(results, axis=-1)
        all_distances = np.concatenate(all_distances, axis=-1)

        label_fn = lambda idx: gram.labels[idx]

        errors = np.zeros_like(all_results.reshape([4, 6]))
        labels = label_fn(all_results.reshape([4, 6]))
        for r in range(labels.shape[0]):
            for c in range(labels.shape[1]):
                if labels[r, c] != labels[r, 0]:
                    errors[r, c] = 1


        def to_tensor(img, error):
            thickness = 10
            img_tensor = t(img)  # type: torch.Tensor
            if error:
                color = torch.tensor([1., 0., 0.])
                x = img_tensor.permute(1, 2, 0)
                x[:thickness, :, :] = color
                x[-thickness:, :, :] = color
                x[:, :thickness, :] = color
                x[:, -thickness:, :] = color
                return x.permute(2, 0, 1)
            return img_tensor


        print('map to tensors')

        # img_tensors = list(map(t, imgs))
        imgs = images[model][all_results.flatten()]
        t = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        img_tensors = [to_tensor(img, err) for img, err in zip(imgs, errors.flatten())]
        print('make grid...')
        grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))

        ax = axes[i]
        ax.imshow(grid)

        ax.set_yticks([])
        # ax.set_title(model, size='large')
        ax.set_ylabel(model, Fontsize=20)
        ax.set_xticks(np.arange(6) * 256 + 128)
        ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'], Fontsize=20)
    fig.set_size_inches(4, 12)
    fig.tight_layout()
    fig.savefig(f'equal_weights_topk.pdf')
    plt.show()
