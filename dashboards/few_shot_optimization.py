import os
import sys

import numpy as np
import plotly.graph_objs as go
import streamlit as st
import torch
import torchvision
from sklearn.utils import shuffle

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import top_k_neighbors, pad_grams, Grams, ImageLoader, dataset_selector, plot, warn_and_get_pca70
from few_shot import optimize

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    data_root, dataset_name, model_name, grams_root, grams_name = dataset_selector(project_root)

    grams = Grams(grams_root)
    num_examples = len(grams.labels)

    os.makedirs(os.path.join(file_dir, 'cache'), exist_ok=True)

    pca_70 = warn_and_get_pca70(file_dir=file_dir,
                                dataset_name=dataset_name,
                                model_name=model_name,
                                grams_name=grams_name,
                                grams=grams)
    graph_files = grams.graph_files
    del grams

    images = ImageLoader(grams_root=grams_root,
                         img_root=os.path.join(data_root, 'imgs'),
                         black_to_white=dataset_name == 'ABC')

    name_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], graph_files))
    }

    positive_vals = {
        'SolidLETTERS': 'y_Abhaya Libre_upper\n'
                        'n_Abhaya Libre_upper',
        'ABC': '3 Fillet - Part 1\n'
               'Part Studio 1 - Corner 8'
    }
    negative_vals = {
        'SolidLETTERS': 'l_Seaweed Script_upper\n'
                        'f_Seaweed Script_upper',
        'ABC': '4 Chamfer - Part 1'
    }

    positives_text = st.sidebar.text_area(label='Enter file names (one per line)',
                                          value=positive_vals[dataset_name] if dataset_name in positive_vals.keys()
                                          else '')
    positives_idx = list(map(lambda n: name_idx[n], positives_text.strip().split('\n')))

    use_random_negatives = st.sidebar.checkbox(label='Use random negatives?',
                                               value=False)
    if use_random_negatives:
        num_negatives = st.sidebar.number_input(label='Number of negatives',
                                                value=100,
                                                step=1,
                                                min_value=0,
                                                max_value=len(graph_files) - len(positives_idx))
        all_idx = set(list(range(len(graph_files))))
        remaining_idx = all_idx.difference(set(positives_idx))
        negatives_idx = list(np.random.choice(len(remaining_idx), num_negatives, replace=False))
    else:
        negatives_text = st.sidebar.text_area(label='Enter file names (one per line)',
                                              value=negative_vals[
                                                  dataset_name] if dataset_name in negative_vals.keys() else '')
        negatives_idx = list(map(lambda n: name_idx[n], negatives_text.strip().split('\n')))

    query_idx = positives_idx[0]

    st.sidebar.subheader('Positive Egs')
    for i in positives_idx:
        st.sidebar.image(images[i], width=200)

    st.sidebar.subheader('Negative Egs')
    for i in negatives_idx:
        st.sidebar.image(images[i], width=200)
    remaining = shuffle(list(set(np.arange(num_examples)).difference(positives_idx)))
    negatives_idx += remaining[:100]

    if pca_70 is not None:
        grams = list(pca_70.values())[:7]
        padded_grams = pad_grams(grams).to(device)

        if st.button('Start Optimization'):
            with st.spinner('optimizing...'):
                optimal_weights = optimize(positive_idx=positives_idx,
                                           negative_idx=negatives_idx,
                                           grams=grams,
                                           metric='cosine')

            combos = {
                'Uniform Layer Weights': torch.ones(7).to(device),
                'Optimized': torch.tensor(optimal_weights).to(device)
            }

            for name, weights in combos.items():
                st.subheader(name)
                print('compute neighbors...')
                neighbors, distances = top_k_neighbors(X=padded_grams,
                                                       weights=weights,
                                                       queries=[query_idx],
                                                       k=6)

                st.subheader('weights')
                st.write(weights)
                st.subheader(f'ids')
                st.write(neighbors)

                print('map to tensors')
                imgs = images[neighbors.flatten()]
                t = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((128, 128)),
                    torchvision.transforms.ToTensor()
                ])
                img_tensors = list(map(t, imgs))
                print('make grid...')
                grid = torchvision.utils.make_grid(img_tensors, nrow=6).permute((1, 2, 0))
                print('make figure...')
                fig = plot(grid_array=grid,
                           query_idx=[query_idx],
                           img_size=128,
                           k=6,
                           distances=distances.flatten())
                print('st plot...')
                st.pyplot(fig)

            st.subheader('Optimized Weights')
            fig = go.Figure(go.Bar(x=np.arange(len(optimal_weights)),
                                   y=optimal_weights))
            st.plotly_chart(fig)
        else:
            st.write('Optimization may take a minute or two.')