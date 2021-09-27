import numpy as np
from occwl.graph import face_adjacency

from occwl.uvgrid import uvgrid


def feature_extractor(solid):
    g = face_adjacency(solid, self_loops=False)

    feats = []
    for face_idx in g.nodes:
        face = g.nodes[face_idx]['face']
        positions = uvgrid(face, num_u=10, num_v=10, method='point')
        normals = uvgrid(face, num_u=10, num_v=10, method='normal')
        mask = uvgrid(face, num_u=10, num_v=10, method='inside')
        feats.append(np.concatenate([positions, normals, mask], axis=-1))
    feats = np.stack(feats)
    max = feats[:, :, :, :3].max(0)
    min = feats[:, :, :, :3].min(0)
    extents = max - min
    feats[:, :, :, :3] = feats[:, :, :, :3] / (extents.max() / 2)
    return feats
