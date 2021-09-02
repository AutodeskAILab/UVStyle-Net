import os
import pickle

import PIL
import dgl
import torch
import pytorch_lightning as pl
import torchvision
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning import Trainer, EvalResult
from torch import nn
from torch.nn import Linear, Sequential, Flatten
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import helper
import parse_util
from networks import brep_model, classifier, nurbs_model
from solid_mnist import SolidMNIST, my_collate, simclr_collate, random_rotate, RandomCrop, identity_transform
import pandas as pd


class FontLabels:
    df = pd.read_csv('analysis/all_fonts_svms/uvnet_labels.csv', sep=',')

    @staticmethod
    def from_file_name(file_name):
        name = file_name.split('_')[1]
        idx = FontLabels.df[FontLabels.df['file_names'] == name].index
        return idx.values


def compute_activation_stats(bg, activations):
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        # F = num faces
        # d = num filters/dimensions
        # graph_activations shape: F x d x 10 x 10
        x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
        x = torch.cat(list(x), dim=-1)  # x shape: d x 100F
        inorm = torch.nn.InstanceNorm1d(x.shape[0])
        x = inorm(x.unsqueeze(0)).squeeze()
        img_size = x.shape[-1]  # img_size = 100F
        gram = torch.matmul(x, x.transpose(0, 1)) / img_size
        grams.append(gram.flatten())
    return torch.stack(grams)


def log_activation_stats(bg, all_layers_activations):
    stats = {layer: compute_activation_stats(bg, layer, activations)
             for layer, activations in all_layers_activations.items()}
    return stats


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, dim=1)


class SimCLR_UVNet(SimCLR):
    def __init__(self,
                 batch_size,
                 num_samples,
                 warmup_epochs=10,
                 lr=1e-4,
                 opt_weight_decay=1e-6,
                 loss_temperature=0.5,
                 model=None,
                 test_mode='tensorboard',
                 log_dir=None,
                 **kwargs):
        super(SimCLR, self).__init__()
        self.save_hyperparameters()
        self.test_mode = test_mode

        self.nt_xent_loss = nt_xent_loss
        self.model = model
        self.encoder = self.init_encoder()

        # h -> || -> z
        self.projection = Projection(input_dim=20)
        self.writer = SummaryWriter(log_dir='tb_log' if log_dir is None else log_dir)

    def init_encoder(self):
        self.model.requires_grad_(False)
        linear = Linear(98340, 20)
        enc = Sequential(
            self.model,
            linear
        )
        return enc

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss

    def test_step(self, batch, batch_idx):
        bg, letter_labels, _, imgs, files = batch
        letter_labels = letter_labels.flatten()
        font_labels = np.concatenate(list(map(FontLabels.from_file_name, files)), axis=0)
        emb = self.encoder(bg).detach().cpu()
        result = EvalResult()
        result.out = letter_labels, torch.tensor(font_labels), emb, files
        return result

    def file_to_image(self, image_path):
        try:
            image = PIL.Image.open(image_path)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([64, 64]),
                torchvision.transforms.ToTensor()
            ])
            image = transform(image)
        except Exception as e:
            print(f'Warning could not load image {image_path} - {e}')
            image = torch.ones([3, 64, 64])
        return image

    def test_epoch_end(
            self, outputs):
        # get image
        embs = []
        letters = []
        fonts = []
        files = []
        for batch in outputs['out']:
            b_letters, b_fonts, b_embs, b_files = batch
            embs.append(b_embs)
            letters.append(b_letters)
            fonts.append(b_fonts)
            files += b_files

        embs = torch.cat(embs).detach().cpu()
        letters = torch.cat(letters).detach().cpu()
        fonts = torch.cat(fonts).detach().cpu()

        if self.test_mode == 'tensorboard':
            img_root = '/home/ubuntu/solid-mnist/jpeg/test/'
            img_paths = map(lambda f: img_root + f[:-4] + '.jpeg', files)
            meta = torch.stack([letters, fonts])
            imgs = torch.stack(list(map(self.file_to_image, img_paths)))
            self.writer.add_embedding(mat=embs,
                                      metadata=meta.transpose(0, 1).tolist(),
                                      label_img=imgs,
                                      metadata_header=['letter', 'font'])

        elif self.test_mode == 'save_embeddings':
            embs = embs.numpy()
            np.save('embeddings', embs)
            np.savetxt('graph_files.txt', np.array(files, dtype=np.str), fmt='%s')
        return EvalResult()



class Model(nn.Module):
    def __init__(self, num_classes, args):
        """
        Model used in this classification experiment
        """
        super(Model, self).__init__()
        self.nurbs_feat_ext = nurbs_model.get_face_model(
            nurbs_model_type=args.nurbs_model_type,
            output_dims=args.nurbs_emb_dim,
            mask_mode=args.mask_mode,
            area_as_channel=args.area_as_channel,
            input_channels=args.input_channels)
        self.brep_feat_ext = brep_model.get_graph_model(
            args.brep_model_type, args.nurbs_emb_dim, args.graph_emb_dim)
        self.cls = classifier.get_classifier(
            args.classifier_type, args.graph_emb_dim, num_classes, args.final_dropout)
        self.nurbs_activations = None
        self.gnn_activations = None

    def forward(self, bg):
        feat = bg.ndata['x'].permute(0, 3, 1, 2)
        out = self.nurbs_feat_ext(feat)
        self.nurbs_activations = self.nurbs_feat_ext.activations
        node_emb, graph_emb = self.brep_feat_ext(bg, out)
        self.gnn_activations = self.brep_feat_ext.activations
        out = self.cls(graph_emb)

        nurb_grams = []
        for layer, activations in self.nurbs_activations.items():
            gram = compute_activation_stats(bg, activations)
            nurb_grams.append(gram)
        self.nurbs_activations = None
        nurb_grams = torch.cat(nurb_grams, dim=-1)

        gnn_grams = []
        for layer, activations in self.gnn_activations.items():
            gram = compute_activation_stats(bg, activations)
            gnn_grams.append(gram)
        self.gnn_activations = None
        gnn_grams = torch.cat(gnn_grams, dim=-1)

        return torch.cat([nurb_grams, gnn_grams], dim=-1)


if __name__ == '__main__':
    device = 'cuda:0'

    parser = parse_util.get_test_parser("UV-Net Classifier Testing Script for Solids")
    parser.add_argument("--apply_square_symmetry", type=float, default=0.0,
                        help="Probability of applying square symmetry transformation to uv-domain")
    args = parser.parse_args()
    state = helper.load_checkpoint(args.state, map_to_cpu=args.no_cuda)
    print('Args used during training:\n', state['args'])
    # Create model and load weights
    state['args'].input_channels = 'xyz_normals'

    model = Model(26, state['args']).to(device)
    model.load_state_dict(state['model'])

    train_dset = SolidMNIST(root_dir='dataset/bin', split='train',
                            transform=identity_transform,
                            crop_func=RandomCrop())
    val_dset = SolidMNIST(root_dir='dataset/bin', split='val',
                          transform=identity_transform,
                          crop_func=RandomCrop())

    sim_clr = SimCLR_UVNet(batch_size=64, num_samples=len(train_dset), model=model)

    train_loader = DataLoader(dataset=train_dset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              collate_fn=simclr_collate)

    val_loader = DataLoader(dataset=val_dset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            collate_fn=simclr_collate)

    trainer = Trainer(gpus=[0],
                      default_root_dir='lightning_logs_crop_only',
                      resume_from_checkpoint='lightning_logs_crop_only/lightning_logs/version_2/checkpoints/epoch=32.ckpt',
                      )
    trainer.fit(model=sim_clr, train_dataloader=train_loader, val_dataloaders=val_loader)

