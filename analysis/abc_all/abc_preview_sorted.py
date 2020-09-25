import pandas as pd
import streamlit as st
import sys
sys.path.append('../../analysis')

from util import OnTheFlyImages

if __name__ == '__main__':
    imgs = OnTheFlyImages(data_root='../uvnet_data/abc_all',
                          img_root='/Users/t_meltp/abc/pngs/all')
    num_nodes = pd.read_csv('../uvnet_data/abc_num_nodes.csv')

    df = pd.DataFrame({
        'path': imgs.img_paths,
        'name': list(map(lambda p: p.split('/')[-1][:-4], imgs.img_paths))
    })
    num_nodes['name'] = num_nodes['graph_file'].apply(lambda p: p.split('/')[-1][:-4])

    merged = pd.merge(df, num_nodes, on='name')
    merged.sort_values('num_nodes', ascending=False, inplace=True)

    start = int(st.sidebar.text_input(label='start:', value='0'))
    num = int(st.sidebar.text_input(label='limit:', value='100'))
    image_size = st.sidebar.slider(label='image size:',
                                   min_value=50,
                                   max_value=1000,
                                   step=1,
                                   value=100)

    idx = merged.index[start:start + num]
    for i in idx:
        st.image(image=imgs[i], format='PNG', width=image_size)
        st.text(merged['name'].loc[i])