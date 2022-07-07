import time
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional
from torch.utils import data
from torch.utils.data.dataset import T_co

from impl_recon.utils import config_io, geometry_utils, image_utils, io_utils

# Sparsification start IDs for training/validataion/test data
SPRSF_START_MAIN_SEED = 1
# Splitting data into training and validation
SPLIT_TRAIN_VAL_SEED = 2
# Sparsification start IDs for internal train/val in AD
SPRSF_START_SHARED_VAL_SEED = 3
# Selection of full-res subset from training data
SELECT_FULL_RES_SUBSET_SEED = 4


class PhaseType(IntEnum):
    """Training, validation and inference."""
    TRAIN = 0
    VAL = 1
    INF = 2


def downsample(image: torch.Tensor, target_spatial_shape: List[int],
               do_threshold: bool) -> torch.Tensor:
    """Downsample given image to given target spatial shape. Optionally threshold values afterwards
    (for labels). Input is supposed to have the channel dimension.
    """
    if list(image.shape[1:]) == target_spatial_shape:
        return image.detach().clone()
    # Fake batch dimension
    image = image.unsqueeze(0)
    # Downsample with averaging. NOT aligning corners makes most sense.
    # noinspection PyArgumentList
    image_lowres = functional.interpolate(  # type: ignore[call-arg]
        image, size=target_spatial_shape, mode='trilinear', align_corners=False,
        recompute_scale_factor=False)
    if do_threshold:
        image_lowres = torch.ge(image_lowres, 0.5).to(image.dtype)
    image_lowres = image_lowres.squeeze(0)
    return image_lowres


def pad(label: torch.Tensor, target_shape: torch.Tensor) -> torch.Tensor:
    """Pad symmetrically except for Z axis, which is padded only in the beginning."""
    curr_shape = torch.tensor(label.shape)
    # Check if padding is actually required
    if torch.equal(target_shape, curr_shape):
        return label

    padding_amount = target_shape - curr_shape
    padding_beginning = torch.ceil(padding_amount / 2).to(torch.int64).tolist()
    padding_end = torch.floor(padding_amount / 2).to(torch.int64).tolist()
    # Pad z axis only in the beginning
    padding_beginning[2] = int(padding_amount[2].item())
    padding_end[2] = 0
    # List with [pad_2_start, pad_2_end, pad_1_start, pad_1_end, pad_0_start, pad_0_end]
    padding: List[int] = []
    for j in reversed(range(label.ndim)):
        padding.append(padding_beginning[j])
        padding.append(padding_end[j])
    label = functional.pad(label, padding)
    return label


def load_volumes(volumes_dir: Path, casenames: List[str]) -> Tuple[List[torch.Tensor],
                                                                   List[torch.Tensor]]:
    """Load axis-aligned volumes with same spacing."""
    label_data = io_utils.load_image_data(volumes_dir, casenames, torch.device('cpu'))
    labels = [x.image for x in label_data]
    trafos = [x.image_trafo for x in label_data]
    if any([not geometry_utils.is_matrix_scaling_and_transform(x) for x in trafos]):
        raise ValueError('Currently only axis-aligned volumes are supported.')
    spacings_np = [np.diagonal(tr)[:3].copy() for tr in trafos]

    # Convert to spacings to tensors
    spacings = [torch.from_numpy(x).to(torch.float32) for x in spacings_np]
    return labels, spacings


def compute_thick_slices(volume: torch.Tensor, axis: int, step_size: int) -> torch.Tensor:
    """Return a new volume (as a copy) with the same dimensions as input, where each slice is an
    average of x neighboring slices from the original volume.
    """
    thick_slices = []
    max_num_slices_below = round(step_size / 2) - 1 if step_size % 2 == 0 else \
        round(step_size / 2 - 0.5)
    for center_slice_id in range(volume.shape[axis]):
        # The end should be exclusive: [start_id, end_id)
        start_id = center_slice_id - max_num_slices_below
        end_id = min(start_id + step_size, volume.shape[axis])
        start_id = max(start_id, 0)  # must be after end computation
        subvolume = torch.index_select(volume, axis, torch.arange(start_id, end_id))
        assert subvolume.shape[axis] == end_id - start_id
        mean_slice = torch.mean(subvolume, dim=axis)
        mean_slice = (mean_slice >= 0.5).to(volume.dtype)
        thick_slices.append(mean_slice)
    volume_thick = torch.stack(thick_slices, axis)
    assert volume_thick.shape == volume.shape
    return volume_thick


def sparsen_volume(volume: torch.Tensor, spacing: torch.Tensor, offset: torch.Tensor, axis: int,
                   slice_step_size: int, slice_start_id: int, use_thick_slices: bool) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a new sparsified dense volume (as a copy), its new spacing and new spatial offset. If
    slice_step_size <= 1, return the original volume (no copy), spacing and spatial offset.
    """
    if use_thick_slices:
        volume = compute_thick_slices(volume, axis, slice_step_size)

    if slice_step_size <= 1:
        return volume, spacing, offset

    slice_ids = torch.arange(slice_start_id, volume.shape[axis], slice_step_size)
    # This creates a copy
    volume_sparse = torch.index_select(volume, axis, slice_ids)
    spacing_sparse = spacing.clone()
    spacing_sparse[axis] *= slice_step_size
    offset_sparse = offset.clone()
    offset_sparse[axis] += slice_start_id * spacing[axis]
    return volume_sparse, spacing_sparse, offset_sparse


def select_fullres_subset(num_cases_total: int, fraction: float) -> List[int]:
    """Generate a list with IDs selected as full-resolution subset."""
    num_cases_actual = round(num_cases_total * fraction)
    if num_cases_actual == 0:
        return []
    gen_subset = torch.Generator().manual_seed(SELECT_FULL_RES_SUBSET_SEED)
    caseids = torch.multinomial(torch.ones(num_cases_total), num_cases_actual, False,
                                generator=gen_subset).tolist()
    return caseids


def split_train_val_caseids(num_cases: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    """Return (reproducible) training and validation case IDs."""
    if num_cases <= 0:
        raise ValueError('Number of cases must be non-negative.')
    val_fraction = max(min(val_fraction, 1.0), 0.0)
    val_set_size = round(num_cases * val_fraction)
    if val_set_size == 0:
        raise ValueError(f'Empty validation set (fraction {val_fraction} of {num_cases} cases).')
    if val_set_size == num_cases:
        raise ValueError(f'Empty training set (fraction {val_fraction} of {num_cases} cases).')

    generator_val_sampling = torch.Generator().manual_seed(SPLIT_TRAIN_VAL_SEED)
    caseids = torch.randperm(num_cases, generator=generator_val_sampling).tolist()
    caseids_train = caseids[:-val_set_size]
    caseids_valid = caseids[-val_set_size:]
    return caseids_train, caseids_valid


def split_train_val_casenames(casenames: List[str], val_fraction: float,
                              is_training: bool) -> List[str]:
    train_caseids, val_caseids = split_train_val_caseids(len(casenames), val_fraction)
    if is_training:
        return [casenames[idx] for idx in train_caseids]
    return [casenames[idx] for idx in val_caseids]


def generate_mid_slice_ids(label: torch.Tensor) -> torch.Tensor:
    """Generate [N, 3] voxels IDs from three orthogonal, center slices."""
    # Find center of foreground gravity
    individual_voxel_ids = [torch.arange(num_elements) for num_elements in label.shape]
    individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
    voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
    fg_positions = voxel_ids * label.unsqueeze(-1)
    fg_positions = fg_positions.reshape(-1, 3)
    center = torch.sum(fg_positions, dim=0) / torch.sum(label)
    center = torch.round(center).to(torch.int64)

    shape = label.shape
    voxel_ids_per_slice = []
    for i in range(3):
        axis1 = (i + 1) % 3
        axis2 = (i + 2) % 3
        individual_voxel_ids = [torch.arange(shape[axis1]), torch.arange(shape[axis2])]
        individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
        individual_voxel_ids_meshed_flat = [torch.flatten(x) for x in individual_voxel_ids_meshed]
        slice_voxel_ids = torch.empty((shape[axis1] * shape[axis2], 3), dtype=torch.int64)
        slice_voxel_ids[:, i] = center[i]
        slice_voxel_ids[:, axis1] = individual_voxel_ids_meshed_flat[0]
        slice_voxel_ids[:, axis2] = individual_voxel_ids_meshed_flat[1]
        voxel_ids_per_slice.append(slice_voxel_ids)
    voxel_ids = torch.cat(voxel_ids_per_slice, 0)
    return voxel_ids


class ReconNetDataset(data.Dataset):
    def __init__(self, labels_dir: Path, casenames: List[str],
                 slice_step_size: int, slice_step_axis: int, use_thick_slices: bool, crop_size: int,
                 do_deterministic_sparsing: bool):
        """
        Dataset for ReconNet.
        :param labels_dir:
        :param casenames: List of casenames to be processed.
        :param slice_step_size: must be at least 2.
        :param slice_step_axis:
        :param crop_size: if 0, disabled.
        :param do_deterministic_sparsing: use deterministic sparsification offset and not random
        """
        self.casenames = casenames
        # Store this for deterministic random sampling operations later
        num_cases_original = len(self.casenames)
        caseids = list(range(num_cases_original))
        self.labels, self.spacings = load_volumes(labels_dir, self.casenames)
        self.offsets = [x / 2 for x in self.spacings]  # type: List[torch.Tensor]

        # Data sanity checks
        if not all(self.labels[0].shape == x.shape for x in self.labels):
            raise ValueError('Equivalent shapes are assumed.')
        if not all(torch.allclose(self.spacings[0], x) for x in self.spacings):
            raise ValueError('Equivalent spacings are assumed.')
        if slice_step_size <= 1:
            raise ValueError('Step size should be at least 2 for ReconNet.')

        self.slice_step_size = slice_step_size
        self.slice_step_axis = slice_step_axis
        self.use_thick_slices = use_thick_slices
        self.crop_size = crop_size

        # Choose deterministic sparsification offsets in advance for whole data or randomly in
        # getitem()
        self.slice_start_ids: Optional[List[int]]
        if do_deterministic_sparsing:
            # Make sure these start IDs are independent of data fraction
            gen = torch.Generator().manual_seed(SPRSF_START_MAIN_SEED)
            self.slice_start_ids = torch.randint(0, slice_step_size, [num_cases_original],
                                                 generator=gen).tolist()
            self.slice_start_ids = [self.slice_start_ids[idx] for idx in caseids]
        else:
            self.slice_start_ids = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item: int):
        label_hr = self.labels[item]

        # Optionally randomly crop
        if self.crop_size > 0:
            assert all(shp >= self.crop_size for shp in label_hr.shape)
            crop_start_x = torch.randint(0, label_hr.shape[0] - self.crop_size + 1, [1]).item()
            crop_start_y = torch.randint(0, label_hr.shape[1] - self.crop_size + 1, [1]).item()
            crop_start_z = torch.randint(0, label_hr.shape[2] - self.crop_size + 1, [1]).item()
            crop_end_x = crop_start_x + self.crop_size
            crop_end_y = crop_start_y + self.crop_size
            crop_end_z = crop_start_z + self.crop_size

            label_hr = label_hr[crop_start_x:crop_end_x,  # type: ignore[misc]
                                crop_start_y:crop_end_y,  # type: ignore[misc]
                                crop_start_z:crop_end_z]  # type: ignore[misc]

        slice_step_axis = self.slice_step_axis
        slice_step_size = self.slice_step_size
        if self.slice_start_ids is None:
            slice_start_id = int(torch.randint(0, slice_step_size, [1]).item())
        else:
            slice_start_id = self.slice_start_ids[item]
        label_lr, spacing_lr, offset_lr = sparsen_volume(
            label_hr, self.spacings[item], self.offsets[item], slice_step_axis, slice_step_size,
            slice_start_id, self.use_thick_slices)
        label_upsampled = image_utils.interpolate_volume(
            label_lr, spacing_lr, offset_lr, torch.tensor(label_hr.shape), self.spacings[item],
            self.offsets[item], 'bilinear')
        # Create channel dim
        label_hr = label_hr.unsqueeze(0)
        label_upsampled = label_upsampled.unsqueeze(0)

        return {
            'labels_lr': label_upsampled,
            'labels': label_hr,
            'spacings': self.spacings[item],
            'casenames': self.casenames[item]
        }


class OrthogonalSlices(data.Dataset):
    def __init__(self, labels_dir: Path, casenames: List[str], verbose: bool):
        self.casenames = casenames
        self.labels, self.spacings = load_volumes(labels_dir, self.casenames)
        # Offsets describe the position of the center of the first voxel. Again, this corresponds to
        # align_corners=False:
        self.offsets = [x / 2 for x in self.spacings]  # type: List[torch.Tensor]

        # Compute max volume size
        # Unlike with typical medical images, we define the bbox size with align_corners=False:
        image_sizes = [torch.tensor(label.shape) * spacing
                       for label, spacing in zip(self.labels, self.spacings)]
        self.image_size: torch.Tensor = torch.stack(image_sizes).max(dim=0)[0]

        if verbose:
            print(f'Volume size: {self.image_size}.')

    def __len__(self):
        return len(self.casenames)

    def __getitem__(self, item: int) -> T_co:
        label = self.labels[item]

        # Extract positions of orthogonal slices in [N, 1, 1, 3]
        voxel_ids = generate_mid_slice_ids(label)
        label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
        voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
        label_values = label_values.unsqueeze(1).unsqueeze(1)

        spacing = self.spacings[item]
        offset = self.offsets[item]
        coords = voxel_ids * spacing + offset

        # Keys are plural because examples will be combined to batches
        result_dict = {
            'coords': coords,
            'labels': label_values,
            'casenames': self.casenames[item],
            'caseids': item,
            'labels_hr': label,
            'spacings_hr': spacing,
            'offsets_hr': offset
        }
        return result_dict


class ImplicitDataset(OrthogonalSlices):
    """For training encoder-free implicit functions. Generated coordinates take spacing into
    account.
    """
    def __init__(self, labels_dir: Path, casenames: List[str],
                 val_fraction: float,  num_points_per_example_per_dim: int, slice_step_size: int,
                 slice_step_axis: int, use_thick_slices: bool,
                 do_yield_full_res: bool, phase_type: PhaseType, verbose: bool):
        """
        Dataset for implicit autodecoder.
        :param labels_dir:
        :param casenames: List of casenames to be processed.
        :param val_fraction: fraction of training data to be used for validation.
        :param num_points_per_example_per_dim: number of random points (per dim) per sparse volume
            to be returned. If -1, then use all points for each example.
        :param slice_step_size: sparsification step size; should be at least 2.
        :param slice_step_axis: sparsification direction.
        :param do_yield_full_res: whether to return full resolution GT in addition to sparse labels.
                                  Should be set to false if batch size != 1 in DataLoader (otherwise
                                  batching won't work).
        :param phase_type: training, validation or inference.
        :param verbose:
        """
        super().__init__(labels_dir, casenames, verbose)
        self.do_yield_full_res = do_yield_full_res

        # This is used for as a link between externally stored latent vectors and data examples
        self.caseids = list(range(len(self.casenames)))

        # Calculate sparse volumes
        if slice_step_size <= 1:
            raise ValueError('Step size should be at least 2.')

        gen = torch.Generator().manual_seed(SPRSF_START_MAIN_SEED)
        slice_start_ids = torch.randint(0, slice_step_size, [len(self.labels)],
                                        generator=gen).tolist()
        results = [sparsen_volume(vol, spacing, offset, slice_step_axis, slice_step_size,
                                  slice_start_id, use_thick_slices)
                   for vol, spacing, offset, slice_start_id in
                   zip(self.labels, self.spacings, self.offsets, slice_start_ids)]
        self.labels_sparse = [x[0] for x in results]
        self.spacings_sparse = [x[1] for x in results]
        self.offsets_sparse = [x[2] for x in results]

        # When training and validating, use some of training examples for both training and
        # validation (with different sparsification starts).
        if phase_type != PhaseType.INF:
            # Reproducible validation set selection
            _, val_caseids = split_train_val_caseids(len(self.casenames), val_fraction)
            # Reproducible sparsification start slice id -- different one for train and val
            gen_val_sparsification = torch.Generator().manual_seed(SPRSF_START_SHARED_VAL_SEED)
            start_slice_ids = torch.randint(0, 2, [len(val_caseids)],
                                            generator=gen_val_sparsification).tolist()
            if phase_type == PhaseType.VAL:
                start_slice_ids = [(x + 1) % 2 for x in start_slice_ids]
            # Sparsify the validation cases
            for caseid, start_slice_id in zip(val_caseids, start_slice_ids):
                self.labels_sparse[caseid], self.spacings_sparse[caseid], \
                    self.offsets_sparse[caseid] = \
                    sparsen_volume(self.labels_sparse[caseid], self.spacings_sparse[caseid],
                                   self.offsets_sparse[caseid], slice_step_axis, 2,
                                   start_slice_id, False)
            # Now drop all non-validation cases for validation -- this HAS to happen after both
            # rounds of sparsification for reproducibility of random generators
            if phase_type == PhaseType.VAL:
                self.labels = [self.labels[i] for i in val_caseids]
                self.spacings = [self.spacings[i] for i in val_caseids]
                self.offsets = [self.offsets[i] for i in val_caseids]
                self.labels_sparse = [self.labels_sparse[i] for i in val_caseids]
                self.spacings_sparse = [self.spacings_sparse[i] for i in val_caseids]
                self.offsets_sparse = [self.offsets_sparse[i] for i in val_caseids]
                self.casenames = [self.casenames[i] for i in val_caseids]
                self.caseids = val_caseids

        self.num_points_per_example = num_points_per_example_per_dim ** self.image_size.numel() \
            if num_points_per_example_per_dim > 0 else -1

    def __len__(self):
        return len(self.casenames)

    def __getitem__(self, item: int):
        label = self.labels_sparse[item]
        # Use sparse (1D) or dense (3D) volumes
        if self.num_points_per_example > 0:  # use random subsample from sparse volume
            # Generate a 1D vector with sparse coords [N, 1, 1, 3]
            ndims = label.ndim
            voxel_ids = torch.empty(self.num_points_per_example, ndims, dtype=torch.int64)
            for i in range(ndims):
                voxel_ids[:, i] = torch.randint(0, label.shape[i], [self.num_points_per_example])
            label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
            voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
            label_values = label_values.unsqueeze(1).unsqueeze(1)
        else:  # use full sparse volume
            # Generate a 3D dense volume [SX, SY, SZ, 3]
            individual_voxel_ids = [torch.arange(num_elements) for num_elements in label.shape]
            individual_voxel_ids_meshed = torch.meshgrid(individual_voxel_ids, indexing='ij')
            voxel_ids = torch.stack(individual_voxel_ids_meshed, -1)
            label_values = label

        spacing = self.spacings_sparse[item]
        offset = self.offsets_sparse[item]
        coords = voxel_ids * spacing + offset

        # Keys are plural because examples will be combined to batches
        result_dict = {
            'coords': coords,
            'labels': label_values,
            'spacings': spacing,
            'offsets': offset,
            'casenames': self.casenames[item],
            'caseids': self.caseids[item]
        }
        # Yield full-res data only during inference (where batch size is 1)
        if self.do_yield_full_res:
            result_dict['labels_hr'] = self.labels[item]
            result_dict['spacings_hr'] = self.spacings[item]
            result_dict['offsets_hr'] = self.offsets[item]
        return result_dict


def create_data_loader(params: Dict, phase_type: PhaseType, verbose: bool) -> data.DataLoader:
    """For AD, use shared training and validation during train/val, and not during inferernce.
    For other tasks, there is no difference between validation and inference.
    """
    if params['labels_dirname'] == 'None':
        raise ValueError('Labels directory is required, it\'s name cannot be \'None\'.')

    base_dir = params['data_basedir']
    labels_dir = base_dir / params['labels_dirname']
    casefiles_basedir = params['casefiles_basedir']

    if not labels_dir.exists() or not casefiles_basedir.exists():
        raise ValueError('At least one of following data directories does not exist:'
                         '\n{}\n{}'
                         .format(labels_dir, casefiles_basedir))

    is_training = phase_type == PhaseType.TRAIN

    # Load the respective casenames
    casefilename = params['train_casefile'] if phase_type != PhaseType.INF \
        else params['test_casefile']
    casenames = io_utils.load_casenames(casefiles_basedir / casefilename)
    # For training & validation, split the data here (except for AD, which does the split itself)
    val_fraction = params['val_fraction']
    if phase_type != PhaseType.INF and params['task_type'] != config_io.TaskType.AD:
        casenames = split_train_val_casenames(casenames, val_fraction, is_training)

    if is_training:
        do_shuffle = True
        data_name = 'training data'
        batch_size = params['batch_size_train']
        num_points_per_example_per_dim = params['num_points_per_example_per_dim_train']
        do_drop_last = True
    else:
        if params['batch_size_val'] != 1:
            print(f'Warning: validation employs full volumes, which may differ in shape between '
                  f'examples. Therefore, running it with batch size {params["batch_size_val"]} > 1 '
                  f'may lead to crashes.')
        do_shuffle = False
        data_name = 'validation data' if phase_type == PhaseType.VAL else 'test data'
        batch_size = params['batch_size_val']
        num_points_per_example_per_dim = -1
        do_drop_last = False
    # Yield full resolution volumes only during inference (with batch size 1, ideally)
    do_yield_full_res = phase_type == PhaseType.INF
    slice_step_size = params['slice_step_size']
    slice_step_axis = params['slice_step_axis']
    use_thick_slices = params['use_thick_slices']
    num_workers = params['num_workers']
    task_type = params['task_type']

    if verbose:
        print(f'Loading {data_name} into memory...')
    t0 = time.time()
    ds: Union[ImplicitDataset, OrthogonalSlices, ReconNetDataset]
    if task_type == config_io.TaskType.AD:
        if 'sample_orthogonal_slices' in params and params['sample_orthogonal_slices']:
            ds = OrthogonalSlices(labels_dir, casenames, verbose)
        else:
            ds = ImplicitDataset(
                labels_dir, casenames, val_fraction, num_points_per_example_per_dim,
                slice_step_size, slice_step_axis, use_thick_slices, do_yield_full_res,
                phase_type, verbose)
    elif task_type == config_io.TaskType.RN:
        crop_size = params['crop_size']
        do_deterministic_sparsing = not is_training
        ds = ReconNetDataset(labels_dir, casenames, slice_step_size, slice_step_axis,
                             use_thick_slices, crop_size, do_deterministic_sparsing)
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    t1 = time.time()
    if verbose:
        print(f'Loading {data_name} done ({len(ds)} images): {t1 - t0:.2f}s')

    dl = data.DataLoader(ds, batch_size, do_shuffle, num_workers=num_workers,
                         drop_last=do_drop_last)
    return dl
