import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage import measure
from scipy import ndimage as ndi
from monai.transforms import MapTransform
import torch

class BodyMaskd(MapTransform):
    """thresholding, erosion, dilation algo to segment the body"""
    def __init__(self, keys, bg_value=-1500) -> None:
        self.keys = keys
        self.bg_value = bg_value
    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            tensor = data[key]
            img = tensor[0].numpy()
            img_g = ndi.gaussian_filter(img, sigma=1.5) # gaussian convolv improves algo for noisy inputs
            body = (img_g >= -500)  # & (I<=win_max)

            if np.sum(body) == 0:
                raise ValueError('body could not be extracted!')

            # Find largest connected component in 3D
            struct = np.ones((3, 3, 3), dtype=np.bool)
            body = ndi.binary_erosion(body, structure=struct, iterations=2)

            body_labels = measure.label(np.asarray(body, dtype=int))

            props = measure.regionprops(body_labels)
            areas = []
            for prop in props:
                areas.append(prop.area)

            # only keep largest, dilate again and fill holes
            body = ndi.binary_dilation(body_labels == (np.argmax(areas) + 1), structure=struct, iterations=2)
            # Fill holes slice-wise
            for z in range(0, body.shape[2]):
                body[:, :, z] = ndi.binary_fill_holes(body[:, :, z])
            for y in range(0, body.shape[1]):
                body[:, y, :] = ndi.binary_fill_holes(body[:, y, :])
            for x in range(0, body.shape[0]):
                body[x, :, :] = ndi.binary_fill_holes(body[x, :, :])

            body = body.astype(np.int8)
            img = np.where(body==0, self.bg_value, img)
            tensor[0] = torch.tensor(img)
            data[key] = tensor
        return data

def nearest_label_filling(img, cc):
    """
    Motivation: Since finding CCs is a reductionary operation, previously labeled voxels may loose labels. The nearest lable filling algorithm label voxels in the lung field that were labeled before cc reduction. implementation:
    1. find signed distance transform of each lobe, where more negative is inside the segmentation and more positive is outside
    2. subtract the binary pre-cc image from the binary post-cc image to find voxels that lost labels
    3. for each such voxel, assign it the label that corresponds to the smallest dt value across all lobes
    4. find cc of filled segmentation - this will remove voxels in the background that were labeled
    """
    dst_no_labels = np.zeros((5, *img.shape))
    no_label = np.where(img, 1, 0) - np.where(cc, 1, 0)
    for i in range(5):
        label = i+1
        binary = np.where(cc==label, 1, 0)
        inv_binary = np.where(cc==label, 0, 1)
        dst = -ndi.distance_transform_cdt(binary) + ndi.distance_transform_cdt(inv_binary)
        dst_no_labels[i, :,:,:] = np.where(no_label, dst, 0)

    nearest = np.argmin(dst_no_labels, axis=0)
    nearest = np.where(no_label, nearest + 1, 0)
    filled = cc + nearest
    return get_largest_cc(filled)

def get_largest_cc(img, connectivity=1):
    """
    :param binaryImg: binary 3d float array
    :param connectivity: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    """
    merged = np.zeros(img.shape)
    for i in range(5):
        label = i+1
        binary = np.where(img==label, 1, 0)
        labels = measure.label(binary, connectivity=connectivity)
        largest_cc = labels==np.argmax(np.bincount(labels.flat, weights=binary.flat))
        merged += largest_cc*label
    return merged

def clip_mask_overlay(in_img, in_mask, out_png, clip_plane='axial', img_vrange=(-1000,600), dim_x=4, dim_y=4):
    '''Plots .png of x by y clips of image volume overlayed with a mask'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image(in_mask, clip_plane, num_clip, idx_clip)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    plot_mask_overlay(clip_in_img_list, 
        clip_mask_img_list, 
        out_png,
        img_vrange=img_vrange,
        mask_vrange=(np.min(in_mask), np.max(in_mask)),
        dim_x=4,
        dim_y=4)

def plot_mask_overlay(clip_in_img_list, clip_mask_img_list, out_png, img_vrange=(-1000,600), mask_vrange=(0,5), dim_x=4, dim_y=4):
    '''Plots overlay from input list of clipped npy'''
    in_img_row_list = []
    mask_img_row_list = []
    for idx_row in range(dim_y):
        in_img_block_list = []
        mask_img_block_list = []
        for idx_column in range(dim_x):
            in_img_block_list.append(clip_in_img_list[idx_column + dim_x * idx_row])
            mask_img_block_list.append(clip_mask_img_list[idx_column + dim_x * idx_row])
        in_img_row = np.concatenate(in_img_block_list, axis=1)
        mask_img_row = np.concatenate(mask_img_block_list, axis=1)
        in_img_row_list.append(in_img_row)
        mask_img_row_list.append(mask_img_row)

    in_img_plot = np.concatenate(in_img_row_list, axis=0)
    mask_img_plot = np.concatenate(mask_img_row_list, axis=0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        in_img_plot,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=img_vrange[0], vmax=img_vrange[1]),
        alpha=0.8)
    ax.imshow(
        mask_img_plot,
        interpolation='none',
        cmap='jet',
        norm=colors.Normalize(vmin=mask_vrange[0], vmax=mask_vrange[1]),
        alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def _clip_image_RAS(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[-clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    else:
        raise NotImplementedError

    return clip

def _clip_image(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
    else:
        raise NotImplementedError

    return clip