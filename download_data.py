import os
import os.path as osp
from argparse import ArgumentParser

import requests
from tqdm import tqdm
from pyunpack import Archive
import shutil


class RemoteItem:
    def __init__(self, name, url, path_maps, temp_dir='temp') -> None:
        super().__init__()
        self.name = name
        self.url = url
        self.path_maps = path_maps
        self.temp_dir = temp_dir

    def _download(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        target = osp.join(self.temp_dir, self.name)
        if os.path.exists(target):
            print(f'File {target} exists - skipping download.')
            return

        print(f'Downloading {self.name}...')
        # Streaming, so we can iterate over the response.
        response = requests.get(self.url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(osp.join(self.temp_dir, self.name), 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    def _extract(self):
        print(f'Extracting {self.name} (this may take some time)...')
        Archive(osp.join(self.temp_dir, self.name)).extractall(self.temp_dir)

    def _move_to_destination(self):
        for src, dest in self.path_maps:
            print(f'Save to {dest}...')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(osp.join(self.temp_dir, src), dest)

    def _cleanup(self):
        print(f'Clean up {self.temp_dir}...')
        shutil.rmtree(self.temp_dir)

    def download_and_extract(self, cleanup=True):
        if all([osp.exists(dest) for src, dest in self.path_maps]):
            print(f'{self.name} already downloaded - skipping.')
            return

        self._download()
        self._extract()
        self._move_to_destination()
        if cleanup:
            self._cleanup()


trained_models = [
    RemoteItem(name='trained_models.7z',
               url='https://0290-mint-brep-style-grams.s3.us-west-2.amazonaws.com/trained_models.7z',
               path_maps=[
                   (osp.join('abc', 'uvnet', 'best.pt'), osp.join('checkpoints', 'uvnet_abc_chkpt.pt')),
                   (osp.join('solidmnist', 'uvnet', 'best_0.pt'),
                    osp.join('checkpoints', 'uvnet_solidletters_chkpt.pt'))
               ]),
]

solid_letters_quickstart = [
    RemoteItem(name='solid_letters_mesh_test.7z',
               url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/mesh_test_clean_extracted.7z',
               path_maps=[
                   ('test_clean_extracted', osp.join('data', 'SolidLETTERS', 'mesh', 'test'))
               ]),
    RemoteItem(name='solid_letters_dgl_test.7z',
               url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/bin_test_clean.7z',
               path_maps=[
                   ('test', osp.join('data', 'SolidLETTERS', 'bin', 'test'))
               ]),
    RemoteItem(name='solid_letters_pngs_test.zip',
               url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/pngs_test_clean.zip',
               path_maps=[
                   ('test_pngs', osp.join('data', 'SolidLETTERS', 'imgs'))
               ]),
    RemoteItem(name='solid_letters_uvnet_grams_all.7z',
               url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_all_sub_mu.7z',
               path_maps=[
                   ('solidmnist_all_sub_mu', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'all'))
               ]),
    RemoteItem(name='solid_letters_uvnet_grams_subset.7z',
               url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_sub_mu_only.7z',
               path_maps=[
                   ('solidmnist_sub_mu_only', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'subset'))
               ]),
]

solid_letters = {
    'grams': [
        RemoteItem(name='solid_letters_uvnet_grams_all.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_all_sub_mu.7z',
                   path_maps=[
                       ('solidmnist_all_sub_mu', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'all'))
                   ]),
        RemoteItem(name='solid_letters_uvnet_grams_subset.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_sub_mu_only.7z',
                   path_maps=[
                       ('solidmnist_sub_mu_only', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'subset'))
                   ]),
        RemoteItem(name='solid_letters_uvnet_grams_all_fnorm.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_all_fnorm.7z',
                   path_maps=[
                       ('solidmnist_all_fnorm', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'all_fnorm_only'))
                   ]),
        RemoteItem(name='solid_letters_uvnet_grams_all_inorm.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/solidmnist_all_inorm.7z',
                   path_maps=[
                       ('solidmnist_all_inorm', osp.join('data', 'SolidLETTERS', 'uvnet_grams', 'all_inorm_only'))
                   ]),
    ],
    'dgl': [
        RemoteItem(name='solid_letters_dgl_test.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/bin_test_clean.7z',
                   path_maps=[
                       ('test', osp.join('data', 'SolidLETTERS', 'bin', 'test'))
                   ]),
        RemoteItem(name='solid_letters_dgl_train.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/bin_train.7z',
                   path_maps=[
                       ('train', osp.join('data', 'SolidLETTERS', 'bin', 'train'))
                   ]),
    ],
    'mesh': [
        RemoteItem(name='solid_letters_mesh_test.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/mesh_test_clean_extracted.7z',
                   path_maps=[
                       ('test_clean_extracted', osp.join('data', 'SolidLETTERS', 'mesh', 'test'))
                   ]),
        RemoteItem(name='solid_letters_mesh_train.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/mesh_train.7z',
                   path_maps=[
                       ('train', osp.join('data', 'SolidLETTERS', 'mesh', 'train'))
                   ]),
    ],
    'pc': [
        RemoteItem(name='solid_letters_pc_test.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/pc_test_clean.7z',
                   path_maps=[
                       ('test', osp.join('data', 'SolidLETTERS', 'pointclouds', 'test'))
                   ]),
        RemoteItem(name='solid_letters_pc_train.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/pc_train.7z',
                   path_maps=[
                       ('train', osp.join('data', 'SolidLETTERS', 'pointclouds', 'train'))
                   ]),
    ],
    'smt': [
        RemoteItem(name='solid_letters_smt_test.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/smt_test_clean.7z',
                   path_maps=[
                       ('test', osp.join('data', 'SolidLETTERS', 'smt', 'test'))
                   ]),
        RemoteItem(name='solid_letters_smt_train.7z',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/smt_train.zip',
                   path_maps=[
                       ('train', osp.join('data', 'SolidLETTERS', 'smt', 'train'))
                   ]),
    ],
    'pngs': [
        RemoteItem(name='solid_letters_pngs_test.zip',
                   url='https://0290-mint-solid-mnist.s3.us-west-2.amazonaws.com/pngs_test_clean.zip',
                   path_maps=[
                       ('test_pngs', osp.join('data', 'SolidLETTERS', 'imgs'))
                   ]),
    ]
}

abc = {
    'grams': [
        RemoteItem(name='abc_grams_all.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/uvnet_abc_sub_mu_only.7z',
                   path_maps=[
                       ('abc_sub_mu_only', osp.join('data', 'ABC', 'uvnet_grams', 'all'))
                   ]),
        RemoteItem(name='pnset_abc_grams_all.7z',
                   url='https://uvstylenet-grams.s3.us-west-2.amazonaws.com/psnet_abc_all.7z',
                   path_maps=[
                       ('abc_all', osp.join('data', 'ABC', 'psnet_grams', 'all'))
                   ])
    ],
    'dgl': [
        RemoteItem(name='abc_bin.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/bin.7z',
                   path_maps=[
                       ('bin', osp.join('data', 'ABC', 'bin'))
                   ])
    ],
    'mesh': [
        RemoteItem(name='abc_mesh.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/new_obj.zip',
                   path_maps=[
                       ('new_obj', osp.join('data', 'ABC', 'mesh'))
                   ])
    ],
    'pc': [
        RemoteItem(name='abc_pc.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/pointclouds.7z',
                   path_maps=[
                       ('pointclouds', osp.join('data', 'ABC', 'pointclouds'))
                   ])
    ],
    'smt': [
        RemoteItem(name='abc_smt.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/smb.7z',
                   path_maps=[
                       ('smb', osp.join('data', 'ABC', 'smt'))
                   ])
    ],
    'pngs': [
        RemoteItem(name='abc_pngs.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/pngs.7z',
                   path_maps=[
                       ('pngs', osp.join('data', 'ABC', 'imgs'))
                   ])
    ],
    'labels': [
        RemoteItem(name='abc_labels.7z',
                   url='https://0290-mint-abc-uvnet.s3.us-west-2.amazonaws.com/abc_labels.7z',
                   path_maps=[
                       ('abc_labels', osp.join('data', 'ABC', 'subset_labels'))
                   ])
    ]
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('selection', choices=['quickstart', 'models', 'solid_letters', 'abc'])
    shared = parser.add_argument_group('solid_letters, abc')
    abc_group = parser.add_argument_group('abc only')
    shared.add_argument('--all', action='store_true', default=False)
    shared.add_argument('--grams', action='store_true', default=False)
    shared.add_argument('--dgl', action='store_true', default=False)
    shared.add_argument('--mesh', action='store_true', default=False)
    shared.add_argument('--pc', action='store_true', default=False)
    shared.add_argument('--smt', action='store_true', default=False)
    shared.add_argument('--pngs', action='store_true', default=False)
    abc_group.add_argument('--labels', action='store_true', default=False)
    args = parser.parse_args()

    if args.selection == 'quickstart':
        items = trained_models + solid_letters_quickstart
    elif args.selection == 'models':
        items = trained_models
    elif args.selection in ['solid_letters', 'abc']:
        if args.all:
            args.grams = True
            args.dgl = True
            args.mesh = True
            args.pc = True
            args.smt = True
            args.pngs = True
            args.labels = True

        data = {
            'solid_letters': solid_letters,
            'abc': abc
        }
        items = []
        if args.grams:
            items += data[args.selection]['grams']
        if args.dgl:
            items += data[args.selection]['dgl']
        if args.mesh:
            items += data[args.selection]['mesh']
        if args.pc:
            items += data[args.selection]['pc']
        if args.smt:
            items += data[args.selection]['smt']
        if args.pngs:
            items += data[args.selection]['pngs']
        if args.labels:
            if 'labels' in data[args.selection].keys():
                items += data[args.selection]['labels']

    else:
        raise 'selection argument must be in [quickstart, models, solid_letters, abc]'

    for item in items:
        item.download_and_extract()
        print(f'Completed {item.name}.\n')
