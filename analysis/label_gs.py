import numpy as np
import pandas as pd
import streamlit as st

from util import Images, IdMap, Grams

if __name__ == '__main__':
    grams = Grams(data_root='uvnet_data/solidmnist_single_letter')
    images = Images(data_root='uvnet_data/solidmnist_single_letter',
                    img_root='/Users/t_meltp/uvnet-img/png/test',
                    cache_file='cache/single_letter_pngs')

    styles = ['slanted', 'serif', 'no curve']
    labels = []
    for i, img in enumerate(images):
        st.image(img.resize((128, 128)))
        labels.append({
            style: st.checkbox(label=style, key=f'{i}_{style}') for style in styles
        })
    button = st.button('Process')
    if button:
        df = pd.DataFrame(labels)
        df['uvnet_id'] = df.index
        id_map = IdMap(src_file='uvnet_data/solidmnist_single_letter/graph_files.txt',
                       dest_file='pointnet2_data/solidmnist_single_letter/graph_files.txt')
        df['pointnet_id'] = id_map(df.index.to_list())
        df.to_csv('extra_labels.csv', index=False)