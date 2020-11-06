import math
import os
from glob import glob
from pathlib import Path

import numpy as np
import trimesh
from joblib import Parallel, delayed
from tqdm import tqdm


def compute(file):
    out_file = output_meshes_path + '/' + Path(file).stem + '.png'
    if os.path.exists(out_file):
        return
    mesh = trimesh.load_mesh(file)
    rotate = trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])
    rotate2 = trimesh.transformations.rotation_matrix(-math.pi / 8, [0, 1, 0])
    scene = mesh.scene()
    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(rotate, camera_old)
    camera_new = np.dot(rotate2, camera_new)
    camera_new = np.dot(trimesh.transformations.translation_matrix([-.3, 0, .8]), camera_new)
    scene.graph[scene.camera.name] = camera_new
    png = scene.save_image(resolution=[128, 128], visible=True)
    with open(out_file, 'wb') as f:
        f.write(png)
        f.close()


if __name__ == '__main__':
    input_meshes_path = '/home/pete/brep_style/solidmnist/mesh/test'
    output_meshes_path = '/home/pete/brep_style/solidmnist/new_pngs'
    if not os.path.exists(output_meshes_path):
        os.makedirs(output_meshes_path)
    files = glob(input_meshes_path + '/*.stl')
    input = tqdm(files)
    Parallel(1)(delayed(compute)(file) for file in input)