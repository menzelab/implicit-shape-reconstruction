import re
from pathlib import Path

import nibabel as nib
import numpy as np


def find_closest_image_axes(ijk_to_lps: np.ndarray) -> tuple[list[int], list[int]]:
    """For each global axis, find a closest image axis."""
    # To get the inverse transformation, remove the scaling to obtain the rotation matrix by
    # normalizing each column. Here we assume that there is no shearing involved (use polar
    # decomposition otherwise). Technically speaking it's an "improper rotation", potentially
    # including reflections (which is what we want here).
    ijk_to_lps = ijk_to_lps / np.linalg.norm(ijk_to_lps, axis=0)
    # Inverted matrix translates global coordinates into image coordinates.
    lps_to_ijk = ijk_to_lps.T
    # Each column this matrix corresponds to a transformed global axis x, y, z in image coordinates.
    # For each transformed global axis, we want to find the closest image axis.
    rotated_x = lps_to_ijk[:, 0]
    rotated_y = lps_to_ijk[:, 1]
    rotated_z = lps_to_ijk[:, 2]
    # For each rotated axis, find closest image axis
    closest_axis_to_x = int(np.argmax(np.abs(rotated_x)))
    closest_axis_to_y = int(np.argmax(np.abs(rotated_y)))
    closest_axis_to_z = int(np.argmax(np.abs(rotated_z)))
    if closest_axis_to_x == closest_axis_to_y or closest_axis_to_y == closest_axis_to_z or \
       closest_axis_to_z == closest_axis_to_x:
        raise ValueError(f'Ambiguous closest axes after rotation: '
                         f'{closest_axis_to_x}, {closest_axis_to_y}, {closest_axis_to_z}\n'
                         f'{lps_to_ijk}')
    # Find closest image axis sign
    sign_closest_axis_to_x = -1 if rotated_x[closest_axis_to_x] < 0 else 1
    sign_closest_axis_to_y = -1 if rotated_y[closest_axis_to_y] < 0 else 1
    sign_closest_axis_to_z = -1 if rotated_z[closest_axis_to_z] < 0 else 1

    return [closest_axis_to_x, closest_axis_to_y, closest_axis_to_z], \
        [sign_closest_axis_to_x, sign_closest_axis_to_y, sign_closest_axis_to_z]


def normalize_coordinates(image_data: np.ndarray,
                          ijk_to_ras: np.ndarray,
                          verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Tranpose and/or flip the image array so that it is close to LPS coordinate system without any
    transformation matrix. Return the resulting image array and spacing.
    """
    # Convert the transformation matrix from nibabel's RAS coordinates to DICOM's LPS coordinates
    ijk_to_lps = ijk_to_ras.copy()
    ijk_to_lps[0] *= -1
    ijk_to_lps[1] *= -1
    assert np.allclose(ijk_to_lps[3, 3], 1.0)
    linear = ijk_to_lps[:3, :3]
    # Find the closest axes
    new_axes, axes_signs = find_closest_image_axes(linear)

    # First transpose, then flip
    # Rearrange the columns
    linear_fixed = linear[:, new_axes]
    img_data_fixed = np.transpose(image_data, new_axes)
    # Flip axes
    linear_fixed = linear_fixed @ np.diag(axes_signs)
    for i, sign in enumerate(axes_signs):
        if sign == -1:
            img_data_fixed = np.flip(img_data_fixed, axis=i)

    # Get the scale w.r.t. new axes
    spacing = np.linalg.norm(linear_fixed, axis=0)

    # Sanity check -- rotation matrix
    rotation = linear_fixed / spacing
    det = np.linalg.det(rotation)
    prod_with_t = rotation.T @ rotation
    if not np.isclose(det, 1.0) or not np.allclose(prod_with_t, np.eye(3), atol=1e-5):
        raise ValueError(f'Remaining matrix is not a rotation matrix. Determinant is {det}, '
                         f'R^T @ R:\n{prod_with_t}')

    if verbose:
        print('closest signed image axes:', [ax * sign for ax, sign in zip(new_axes, axes_signs)])
        print(f'shape: {image_data.shape} -> {img_data_fixed.shape}')
        print('spacing:', spacing)

    return img_data_fixed, spacing


def load_nifti(file: Path, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    if not file.exists():
        raise ValueError(f'Nifti file does not exist:\n{file}')
    img = nib.load(file, mmap=False)
    img_data = img.get_fdata(caching='unchanged')
    img_data = img_data.astype(img.get_data_dtype())
    img_data, spacing = normalize_coordinates(img_data, img.affine, verbose)
    img_data = np.ascontiguousarray(img_data)
    return img_data, spacing


def save_nifti(volume: np.ndarray, target_file: Path, spacing: tuple[int]):
    affine_diag = [*spacing, 1.0]
    affine = np.diag(affine_diag)
    # Make sure matrix is stored in RAS instead of LPS (this is what nibabel expects)
    affine[0] *= -1
    affine[1] *= -1
    img = nib.Nifti1Image(volume, affine)
    nib.save(img, target_file)


class DataGeneartor:
    def __init__(self, labels_dir: Path, labels_pattern: str, verbose: bool):
        if not labels_dir.exists():
            raise ValueError(f'Labels directory does not exist:\n{labels_dir}')

        self.label_paths = list(labels_dir.glob(labels_pattern))
        self.verbose = verbose

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray, str]:
        lbl_path = self.label_paths[item]
        casename_match = re.match(r'verse\d{3}', lbl_path.name)
        if casename_match is None:
            raise ValueError(f'Filename does not match with casename pattern:\n{lbl_path.name}')
        casename = casename_match[0]

        lbl, spacing_lbl = load_nifti(lbl_path, self.verbose)

        return lbl, spacing_lbl, casename


def get_fg_size_and_position(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = np.zeros(3, np.int32)
    center_pos = - np.ones(3, np.int32)
    if not np.any(mask):
        # Empty image
        return size, center_pos

    for i in range(3):
        if i == 0:  # sag
            proj = np.max(mask, axis=(1, 2))
        elif i == 1:  # cor
            proj = np.max(mask, axis=(0, 2))
        else:  # ax
            proj = np.max(mask, axis=(0, 1))
        proj_list = proj.tolist()
        assert isinstance(proj_list, list)
        first_fg_pos = -1
        for j, val in enumerate(proj_list):
            if val == 1:
                first_fg_pos = j
                break
        last_fg_pos = -1
        for j, val in enumerate(reversed(proj_list)):
            if val == 1:
                last_fg_pos = len(proj_list) - j
                break
        if first_fg_pos == -1 or last_fg_pos == -1:
            raise ValueError('No ones found -- this shouldn\'t happen!')
        size[i] = last_fg_pos - first_fg_pos
        center_pos[i] = first_fg_pos + round(size[i] / 2)
    return size, center_pos


def crop_and_pad(image: np.ndarray, position: np.ndarray, bbox_size: int) -> np.ndarray:
    start_crop = np.round(position - bbox_size / 2).astype(np.int32)
    end_crop = np.round(position + bbox_size / 2).astype(np.int32)
    start_crop = np.maximum(start_crop, 0)
    end_crop = np.minimum(end_crop, np.array(image.shape))
    crop = image[start_crop[0]:end_crop[0], start_crop[1]:end_crop[1], start_crop[2]:end_crop[2]]
    # Pad to desired bb size
    pad_sizes = bbox_size - np.array(crop.shape)
    pad_beginning = np.floor(pad_sizes / 2).astype(np.int32)
    pad_end = np.ceil(pad_sizes / 2).astype(np.int32)
    crop = np.pad(crop, list(zip(pad_beginning, pad_end)))
    assert np.all(np.equal(crop.shape, bbox_size))
    return crop


def save_individual_vertebrae(labels: np.ndarray, spacing_lbl: np.ndarray, bbox_size: int,
                              target_labels_dir: Path,
                              casename: str) -> tuple[np.ndarray, np.ndarray]:
    """Process single case with multiple vertebra. Return the average and max vertebra size (from
    labels)."""
    max_size = np.zeros(3, np.int32)
    avg_size = np.zeros(3, np.float32)

    vrtbr_types = np.unique(labels)[1:]  # skip background label 0

    vertebra_selection = vrtbr_types >= 20  # lumbar only
    if np.sum(vertebra_selection) == 0:
        return max_size, avg_size

    types_masked = vrtbr_types[vertebra_selection]
    for vrtbr_type in types_masked:
        labels_binary = np.equal(labels, vrtbr_type).astype(np.uint8)
        assert np.any(labels_binary)
        # Extract the vertebra size and position from binary mask
        vertebra_size, position = get_fg_size_and_position(labels_binary)
        max_size = np.maximum(vertebra_size, max_size)
        avg_size += vertebra_size
        if np.any(vertebra_size > bbox_size - 2):
            raise ValueError(f'Vertebra type {vrtbr_type} in case {casename} is larger than '
                             f'bbox - 2: {vertebra_size} vs {bbox_size - 2}.')
        crop_lbl = crop_and_pad(labels_binary, position, bbox_size)
        save_nifti(crop_lbl, target_labels_dir / f'{casename}_{vrtbr_type:02d}.nii.gz',
                   tuple(spacing_lbl))
    return max_size, avg_size / len(types_masked)


def main():
    """Transpose+flip label arrays so that indices (i,j,k) are aligned with anatomical axes (L,P,S).
    Then extract volumes with individual vertebra.
    """
    np.set_printoptions(precision=4, suppress=True)
    labels_dir = Path('/Volumes/Extreme SSD/Data/spine/verse2019/all_data/labels')
    target_labels_dir = Path(
        '/Volumes/Extreme SSD/Data/spine/verse2019/lumbar_centered_isotropic/labels2')
    verbose_loading = False
    pattern = '*.nii*'
    bbox_size = 128

    data_gen = DataGeneartor(labels_dir, pattern, verbose_loading)

    max_size = np.zeros(3, np.int32)
    avg_size = np.zeros(3, np.float32)
    num_avgs = 0
    num_processed = 0
    for i, (labels, spacing_lbl, casename) in enumerate(data_gen):
        print(f'Case {i + 1}/{len(data_gen)}: {casename}')
        # Skip non-isotropic spacings
        if not np.allclose(spacing_lbl, spacing_lbl[0]):
            print('Non-isotropic voxel size, skipping')
            continue
        num_processed += 1
        labels = labels.astype(np.uint8)
        max_curr, avg_curr = save_individual_vertebrae(labels, spacing_lbl, bbox_size,
                                                       target_labels_dir, casename)
        max_size = np.maximum(max_curr, max_size)
        if not np.all(avg_curr == 0.0):
            avg_size += avg_curr
            num_avgs += 1

    print(f'\n\nMax vertebra size: {max_size}\nAvg vertebra size:{avg_size / num_avgs}')
    print(f'Processed {num_processed}/{len(data_gen)} cases.')


if __name__ == '__main__':
    main()
