import numpy as np
from torch.utils.data.dataloader import DataLoader

from transform.data import GramsDataset, collate
from transform.models import AffineLinearModel

if __name__ == '__main__':
    model = AffineLinearModel.load_from_checkpoint(
        checkpoint_path='lightning_logs/version_72/checkpoints/epoch=11.ckpt',
        hparams_file='lightning_logs/version_72/hparams.yaml')

    dset = GramsDataset()

    dataloader = DataLoader(dataset=dset,
                            batch_size=32,
                            shuffle=False,
                            collate_fn=collate)

    embs = []
    ys = []
    graph_files = []
    for i, (x_batch, y_batch, graph_files_batch) in enumerate(dataloader):
        model.eval()
        emb_batch = model.embedding(x_batch)
        embs.append(emb_batch.detach().numpy())
        ys.append(y_batch.detach().numpy())
        graph_files += graph_files_batch

    embs = np.concatenate(embs, axis=0)
    ys = np.concatenate(ys, axis=0)
    np.save('embeddings_1026', embs)
    np.savetxt('labels_1026.txt', ys, fmt='%d')
    np.savetxt('graph_files_1026.txt', graph_files, fmt='%s')