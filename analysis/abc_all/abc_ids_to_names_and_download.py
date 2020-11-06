import os
import subprocess
import sys

import streamlit as st
import numpy as np

if __name__ == '__main__':
    data_root = '../uvnet_data/abc_sub_mu_only/'
    graph_files = np.loadtxt(f'{data_root}/graph_files.txt', dtype=np.str, delimiter='\n')

    text = st.text_area(label='Enter space separated ids:')

    if len(text) > 0:
        idx = list(map(int, text.split(' ')))

        names = st.text_area('line separated names:', value='\n'.join(map(lambda n: n[:-4], graph_files[idx])))
    else:
        names = st.text_area('line separated names:')

    button = st.button('Download')
    target_dir = st.text_input(label='Download to:', value=f'{os.getenv("HOME")}/Downloads')

    if button and len(names) > 0:
        with st.spinner('downloading...'):
            for name in names.split('\n'):
                out = subprocess.run(['scp', '-T',
                                      '-i', f'{os.getenv("HOME")}/.ssh/t_meltp_brep_style.pem',
                                      f'ubuntu@10.55.144.180:"/home/ubuntu/abc/smb/all/{name}".smb',
                                      f'{target_dir}/'], capture_output=True)

                print(out.stdout)
                print(out.stderr, file=sys.stderr)
