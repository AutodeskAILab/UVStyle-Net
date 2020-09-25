import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

import helper
import parse_util
from contrastive import Model, SimCLR_UVNet
from solid_mnist import SolidMNIST, my_collate

if __name__ == '__main__':
    device = 'cuda:0'

    parser = parse_util.get_test_parser("UV-Net Classifier Testing Script for Solids")
    parser.add_argument("--apply_square_symmetry", type=float, default=0.0,
                        help="Probability of applying square symmetry transformation to uv-domain")
    args = parser.parse_args()
    state = helper.load_checkpoint(args.state, map_to_cpu=True)
    print('Args used during training:\n', state['args'])
    # Create model and load weights
    state['args'].input_channels = 'xyz_normals'

    model = Model(26, state['args']).to(device)
    model.load_state_dict(state['model'])

    test_dset = SolidMNIST(root_dir='dataset/bin', split='test')
    test_loader = DataLoader(dataset=test_dset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=8,
                             # sampler=SubsetRandomSampler(np.arange(2000)),
                             pin_memory=True,
                             collate_fn=my_collate)
    sim_clr = SimCLR_UVNet(batch_size=64, num_samples=len(test_dset), model=model, test_mode='tensorboard', log_dir='contrastive_crop_only')
    # sim_clr = SimCLR_UVNet(batch_size=64, num_samples=len(test_dset), model=model, test_mode='save_embeddings')

    trainer = pl.Trainer(gpus=[0])
    trainer.test(model=sim_clr,
                 test_dataloaders=test_loader,
                 ckpt_path='/home/ubuntu/NURBSNet/lightning_logs_crop_only/lightning_logs/version_3/checkpoints/epoch=34.ckpt')

