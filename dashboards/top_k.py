import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams, ImageLoader, get_pca_70, pad_grams, top_k_neighbors


def plot(grid_array, query_idx, img_size, k, distances=None, font_size=24):
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.imshow(grid_array)

    ax.set_xticks(np.arange(k) * (img_size + 2) + (img_size / 2))
    ax.set_xticklabels(['Q'] + list(map(str, np.arange(1, k))), Fontsize=font_size)

    if query_idx is not None and len(query_idx) > 0:
        ax.set_yticks(np.arange(len(query_idx)) * (img_size + 2) + (img_size / 2))
        ax.set_yticklabels(query_idx if query_idx is not None else None, Fontsize=font_size)
    else:
        ax.set_yticks([])

    if distances is not None:
        x = np.arange(k)
        y = np.arange(len(distances))
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        text = list(map(lambda d: f'{d:.2f}', distances.tolist()))

        space = img_size
        for t, pos in zip(text, xy):
            x, y = pos
            ax.annotate(t, ((x * space + .6 * space), (y * space + .9 * space)), color='red', Fontsize=font_size)

    fig.set_size_inches(21, 15)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    st.title('Top-k Queries')
    dataset_name = st.sidebar.selectbox(label='Dataset',
                                        options=os.listdir(os.path.join(project_root, 'data')))
    data_root = os.path.join(project_root, 'data', dataset_name)

    grams_name = st.sidebar.selectbox(label='Grams',
                                      options=os.listdir(os.path.join(data_root, 'grams')))

    uv_net_data_root = os.path.join(data_root, 'grams', grams_name)
    img_path = os.path.join(data_root, 'imgs')

    grams = Grams(data_root=uv_net_data_root)

    images = ImageLoader(data_root=uv_net_data_root,
                         img_root=img_path,
                         black_to_white=dataset_name == 'ABC')

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams.graph_files))
    }

    name_vals = {
        'SolidLETTERS': 'y_Abhaya Libre_upper\n'
                        'n_Zhi Mang Xing_lower\n'
                        'l_Seaweed Script_upper\n'
                        'e_Turret Road_upper',
        'ABC': '\'16Regs - 2016RegulationBox\n'
               '1_10th_scale_on_road_car Materials v5 v1 v0 parts - Part 121\n'
               '2 - Part 1-8g5pl30u\n'
               '3 Fillet - Part 1\n'
               'Lego - Part 1'
    }
    text_idx_names = st.sidebar.text_area(label='Enter file names (one per line)',
                                          value=name_vals[dataset_name] if dataset_name in name_vals.keys() else '')
    query_idx = list(map(lambda n: name_idx[n], text_idx_names.split('\n')))

    k = st.sidebar.number_input(label='k',
                            value=5,
                            step=1,
                            min_value=1,
                            max_value=10)

    st.sidebar.subheader('Layer Weights')
    defaults = [1., 1., 1., 1., 0., 0., 0.]
    weights = [st.sidebar.slider(label=str(i),
                                 min_value=0.,
                                 max_value=1.,
                                 step=0.01,
                                 value=defaults[i])
               for i in range(len(defaults))]
    weight_combos = torch.tensor([weights], device=device)

    pca_70 = None
    os.makedirs(os.path.join(file_dir, 'cache'), exist_ok=True)
    cache_file = os.path.join(file_dir, 'cache', f'{dataset_name}-{grams_name}-pca_70')
    if not os.path.exists(cache_file):
        st.write('To visualize the query results, you will first need to perform PCA on the Gram matrices.')
        st.write('You will need to do this only once.')
        st.write('For SolidLETTERS this should take less than a minute, and require no more then 1GB memory.')
        st.write('For ABC this will require approx. 36GB memory, and will take a few minutes.')
        if st.button(label='Start'):
            pca_70 = get_pca_70(grams=grams,
                                cache_file=cache_file,
                                verbose=True)
            del grams
    else:
        pca_70 = get_pca_70(grams=grams,
                            cache_file=cache_file,
                            verbose=True)
        del grams

    if pca_70 is not None:
        padded_grams = pad_grams(list(pca_70.values())).to(device)

        for layer, weights in enumerate(weight_combos):
            print('compute neighbors...')
            neighbors, distances = top_k_neighbors(X=padded_grams,
                                                   weights=weights,
                                                   queries=query_idx,
                                                   k=k + 1,
                                                   metric='cosine')

            print('map to tensors')
            imgs = images[neighbors.flatten()]
            t = torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor()
            ])
            img_tensors = list(map(t, imgs))
            print('make grid...')
            grid = torchvision.utils.make_grid(img_tensors, nrow=k + 1).permute((1, 2, 0))
            print('make figure...')
            fig = plot(grid_array=grid,
                       query_idx=[chr(65 + i) for i in range(len(query_idx))],
                       img_size=128,
                       k=k + 1,
                       distances=distances.flatten(),
                       font_size=36)
            print('st plot...')
            st.pyplot(fig)
