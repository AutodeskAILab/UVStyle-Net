import os
import sys

import streamlit as st
import torch
import torchvision

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams, ImageLoader, pad_grams, top_k_neighbors, dataset_selector, plot, warn_and_get_pca70

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    st.title('Top-k Queries')

    data_root, dataset_name, grams_root, grams_name = dataset_selector(project_root)
    img_root = os.path.join(data_root, 'imgs')

    grams = Grams(grams_root=grams_root)

    images = ImageLoader(grams_root=grams_root,
                         img_root=img_root,
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

    pca_70 = warn_and_get_pca70(file_dir=file_dir,
                                dataset_name=dataset_name,
                                grams_name=grams_name,
                                grams=grams)
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
