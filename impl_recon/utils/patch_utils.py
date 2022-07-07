from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

"""
This code is adapted from the nnUNet: https://github.com/MIC-DKFZ/nnUNet
"""


def compute_steps_for_sliding_window(patch_size: List[int], image_size: Tuple[int],
                                     step_size: float) -> List[List[int]]:
    """For each dimension, return positions of windows."""
    if any(x < y for x, y in zip(image_size, patch_size)):
        raise ValueError('Image size cannot be smaller than patch size in any dimension. Reduce '
                         'the patch size.')
    if not 0 < step_size <= 1:
        raise ValueError('Step size must be in (0, 1].')

    steps = []
    for img_size, ptch_size in zip(image_size, patch_size):
        # Our step width is patch_size*step_size at most, but can be narrower. For example if we
        # have image size of 110, patch size of 64 and step_size of 0.5, then we want to make 3
        # steps starting at coordinate 0, 23, 46
        target_step_size_in_voxels = ptch_size * step_size
        num_steps = int(np.ceil((img_size - ptch_size) / target_step_size_in_voxels)) + 1
        if num_steps > 1:
            actual_step_size = (img_size - ptch_size) / (num_steps - 1)
        else:  # num_steps == 1, cannot be < 1
            actual_step_size = 0  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps)]
        steps.append(steps_here)

    return steps


def get_gaussian(patch_size: List[int], sigma_scale: float = 0.125) -> np.ndarray:
    """Compute a gaussian weight map."""
    onehot = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    onehot[tuple(center_coords)] = 1
    sigmas = [x * sigma_scale for x in patch_size]
    gaussian_importance_map = gaussian_filter(onehot, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # Gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def predict_tiled(data: torch.Tensor, net: torch.nn.Module, patch_size: List[int]):
    """If patch size is larger than the input image in all dimensions, do a straight-forward
    prediction. Otherwise, do a tiled prediction with gaussian weighting on overlaps.
    """
    if all(x <= y for x, y in zip(data.shape[2:], patch_size)):
        return net(data)

    # Compute the steps for sliding window
    steps = compute_steps_for_sliding_window(patch_size, data.shape[2:], 0.5)
    gaussian_importance_map = get_gaussian(patch_size, 0.125)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).to(data.device)

    # Do the predictions
    aggregated_results = torch.zeros([data.shape[0], 1, *data.shape[2:]], dtype=torch.float32,
                                     device=data.device)
    aggregated_nb_of_predictions = torch.zeros_like(aggregated_results)
    for i in range(data.shape[0]):
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    input_patch = data[i, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z].unsqueeze(0)
                    predicted_patch = net(input_patch)
                    predicted_patch *= gaussian_importance_map

                    aggregated_results[i, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch[0]
                    aggregated_nb_of_predictions[i, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] \
                        += gaussian_importance_map

    # Normalize the aggregated logits
    logits = aggregated_results / aggregated_nb_of_predictions
    return logits
