import numpy as np
import torch
import matplotlib.pyplot as plt

INVALID_FONTS = [
"Bokor",
"Lao Muang Khong",
"Lao Sans Pro",
"MS Outlook",
"Catamaran Black",
"Dubai",
"HoloLens MDL2 Assets",
"Lao Muang Don",
"Oxanium Medium",
"Rounded Mplus 1c",
"Moul Pali",
"Noto Sans Tamil",
"Webdings",
"Armata",
"Koulen",
"Yinmar",
"Ponnala",
"Noto Sans Tamil",
"Chenla",
"Lohit Devanagari",
"Metal",
"MS Office Symbol",
"Cormorant Garamond Medium",
"Chiller",
"Give You Glory",
"Hind Vadodara Light",
"Libre Barcode 39 Extended",
"Myanmar Sans Pro",
"Scheherazade",
"Segoe MDL2 Assets",
"Siemreap",
"Taprom",
"Times New Roman TUR",
"Playfair Display SC Black",
"Poppins Thin",
"Raleway Dots",
"Raleway Thin",
"Segoe MDL2 Assets",
"Segoe MDL2 Assets",
"Spectral SC ExtraLight",
"Txt",
"Uchen",
"Yinmar",
"Almarai ExtraBold",
"Fasthand",
"Exo",
"Freckle Face",
"Montserrat Light",
"Inter",
"MS Reference Specialty",
"MS Outlook",
"Preah Vihear",
"Sitara",
"Barkerville Old Face",
"Bodoni MT"
"Bokor",
"Fasthand",
"HoloLens MDL2 Assests",
"Libre Barcode 39",
"Lohit Tamil",
"Marlett",
"MS outlook",
"MS office Symbol Semilight",
"MS office symbol regular",
"Ms office symbol extralight",
"Ms Reference speciality",
"Segoe MDL2 Assets",
"Siemreap",
"Sitara",
"Symbol",
"Wingdings",
"Metal",
"Ponnala",
"Webdings",
"Aguafina Script"]


def valid_font(filename):
    for name in INVALID_FONTS:
        if name.lower() in str(filename).lower():
            return False
    return True


def save_feature_to_csv(feat: torch.Tensor, filename):
    """
    Save loaded feature to csv file to visualize sampled points
    :param feat Features loaded from *.feat file of shape [#faces, #u, #v, 10]
    :param filename Output csv filename
    """
    assert len(feat.shape) == 4  # faces x #u x #v x 10
    pts = feat[:, :, :, :3].numpy().reshape((-1, 3))
    mask = feat[:, :, :, 6].numpy().reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()
    np.savetxt(filename, pts, delimiter=",", header="x,y,z")


def bounding_box_uvsolid(inp: torch.Tensor):
    pts = inp[:, :, :, :3].reshape((-1, 3))
    mask = inp[:, :, :, 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvsolid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvsolid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[:, :, :, :3] -= center
    inp[:, :, :, :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def center_and_scale_pointcloud(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_pointcloud(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[:, :3] -= center
    inp[:, :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp
