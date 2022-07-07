import sys
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, OrderedDict, Tuple

import nibabel as nib
import numpy as np
import torch

from impl_recon.utils import geometry_utils


@dataclass
class ImageData:
    """Volumetric image or label data corresponding to a single case."""
    casename: str
    image: torch.Tensor
    # Affine transformation matrix that maps tensor indices (each within [0, shape - 1]) to
    # positions of voxel's centers in global coordinates.
    image_trafo: np.ndarray


def save_nifti_file(image_data: np.ndarray, local_to_global: np.ndarray, target_filepath: Path):
    """Save the given numpy array into a given nifti file path."""
    local_to_global_ras = local_to_global.copy()
    # Convert this library's LPS+ coordinate system to nibabel's RAS+ coordinates
    local_to_global_ras[0] *= -1
    local_to_global_ras[1] *= -1
    # Create header in an Amira-compatible way
    header = nib.Nifti1Header()
    header.set_data_dtype(image_data.dtype)
    header.set_qform(local_to_global_ras, code='scanner')
    img = nib.Nifti1Image(image_data, None, header)
    nib.save(img, target_filepath)


def load_nifti_file(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return the image numpy array as well as the local to global affine transformation matrix.
    In the local image coordinate system, the array indices correspond to positions of voxels'
    centers. The transformation matrix therefore translates array indices into cartesian coordinates
    of voxels' centers in the LPS+ coordinate system.
    """
    if not image_path.exists():
        raise ValueError('Nifti file does not exist:\n{}'.format(image_path))
    img = nib.load(str(image_path), mmap=False)
    img_data = img.get_fdata(caching='unchanged')
    img_data = np.ascontiguousarray(img_data).astype(img.get_data_dtype())
    # Convert the transformation matrix from nibabel's RAS coordinates to DICOM's LPS coordinates
    ijk_to_lps = img.affine.copy()
    ijk_to_lps[0] *= -1
    ijk_to_lps[1] *= -1
    return img_data, ijk_to_lps


def load_nifti_files(source_dir: Path, file_pattern: str, casenames: Optional[List[str]] = None) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load nifti files from a given directory that match a given pattern to a list of numpy arrays
    with image data and a list of arrays with transformation matrices. If no casenames list is
    given, load all available files. If casenames list is given, the pattern must contain '{}' for
    the case name.
    The affine transformation matrix maps tensor indices (each within [0, shape - 1]) to positions
    of voxel's centers in global coordinates.
    """
    if casenames is None:
        # Sort the filenames -- the exact algorithm doesn't matter, but the ordering must be
        # consistent
        nifti_files = sorted(source_dir.glob(file_pattern), key=lambda path: path.name)
    else:
        nifti_files = []
        for casename in casenames:
            curr_pattern = file_pattern.format(casename)
            files = list(source_dir.glob(curr_pattern))
            if len(files) != 1:
                raise ValueError('Exactly one file must fit the pattern:\n{}'
                                 .format(source_dir / curr_pattern))
            nifti_files.append(files[0])

    results = [load_nifti_file(nifti_file) for nifti_file in nifti_files]
    return [x[0] for x in results], [x[1] for x in results]


def load_image_data(images_dir: Path, casenames: List[str], target_device: torch.device,
                    file_pattern: str = '{}*.nii*') -> List[ImageData]:
    """
    Load all image data into memory as float32.
    :param images_dir: Directory with image files.
    :param casenames: List of case names.
    :param target_device: Target device for image and labels tensors.
    :param file_pattern: File pattern that includes {} as placeholder for casename.
    """
    images, trafos = load_nifti_files(images_dir, file_pattern, casenames)

    # Currently only transformation matrices with scaling & translation are supported
    if not all([geometry_utils.is_matrix_scaling_and_transform(m) for m in trafos]):
        example_trafos = [str(x) for x in trafos[:5]]
        raise ValueError('Local to global image matrix is supposed to be 4x4, have scaling '
                         'and translation components only, and positive scaling. Here are some '
                         'examples:\n{}'.format('\n'.join(example_trafos)))

    # Convert everything to float32 and/or tensors
    images = [image.astype(np.float32) for image in images]
    images_tensors = [torch.from_numpy(image).to(torch.float32).to(device=target_device)
                      for image in images]
    image_trafos = [trafo.astype(np.float32) for trafo in trafos]

    images_data = [ImageData(casename, image, image_trafo)
                   for casename, image, image_trafo
                   in zip(casenames, images_tensors, image_trafos)]

    return images_data


def load_casenames(filepath: Path) -> List[str]:
    """Load a text file with case names. Empty lines are ignored."""
    if not filepath.exists():
        raise ValueError('Case name file does not exist:\n{}'
                         .format(filepath))
    with open(str(filepath), 'r') as f:
        lines = [line.rstrip() for line in f if line != '\n']

    return lines


class Logger(TextIOWrapper):
    """Log both to a given file as well as stdout. NOTE: delete this object explicitly before
    creating a new one if you want to change the target logging file.
    """
    def __init__(self, filepath: Path, filemode: str):
        super().__init__(sys.__stdout__.buffer)
        self.file = open(filepath, filemode)

    def __del__(self):
        # Do not reset the default stdout so that re-assigning to a new Logger is not overwritten
        self.file.close()

    def write(self, data):
        self.file.write(data)
        sys.__stdout__.write(data)

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()


def find_single_file(source_dir: Path, pattern: str) -> Path:
    """Within a given directory, find a single file that matches a given pattern. Throw exceptions
    if the directory does not exist or contains more than one file that matches the pattern.
    """
    if not source_dir.exists():
        raise ValueError('The source directory does not exist:\n{}'.format(source_dir))

    matched_files = list(source_dir.glob(pattern))
    if len(matched_files) != 1:
        raise ValueError('The source directory has to contain exactly one file that matches the '
                         'pattern {}:\n{}'.format(pattern, source_dir))
    return matched_files[0]


class RollingCheckpointWriter:
    """Writer to write checkpoint files in a rolling fashion (automatically deleting old ones)."""
    def __init__(self, base_dir: Path, base_filename: str, max_num_checkpoints: int,
                 extension: str = 'pth'):
        if not base_dir.exists():
            raise ValueError('Target directory does not exist:\n{}'.format(base_dir))
        if max_num_checkpoints <= 0:
            raise ValueError('Max number of checkpoints must be at least one.')

        self.base_dir = base_dir
        self.base_filename = base_filename
        self.max_num_checkpoints = max_num_checkpoints
        self.extension = extension

    def write_rolling_checkpoint(self, model_state: dict, optimizer_state: dict,
                                 num_steps_trained: int, num_epochs_trained: int):
        """Write the given state into a checkpoint file. Overwrite any existing files. Delete older
        checkpoints in the directory, so that the given upper bound is not exceeded.
        """
        # First, write the checkpoint file
        state = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'num_steps_trained': num_steps_trained,
            'num_epochs_trained': num_epochs_trained,
        }
        target_filepath = self.base_dir / '{}_{}.{}'.format(self.base_filename, num_steps_trained,
                                                            self.extension)
        torch.save(state, target_filepath)

        # Now delete oldest files until we reach the max
        paths = list(self.base_dir.glob('{}_*.{}'.format(self.base_filename, self.extension)))
        num_files_to_delete = len(paths) - self.max_num_checkpoints
        if num_files_to_delete <= 0:
            return
        paths.sort(key=lambda pth: int(pth.stem.split('_')[-1]))
        for path in paths[:num_files_to_delete]:
            path.unlink()


def load_latest_checkpoint(base_dir: Path, base_filename: str,
                           extension: str = 'pth',
                           verbose: bool = False) -> Tuple[OrderedDict, dict, int, int]:
    """
    Load the latest checkpoint from the given directory. Throws an error if the given directory does
    not exist or has no checkpoints.

    :param base_dir: Base model directory.
    :param base_filename: The base name within checkpoint files.
    :param extension: Checkpoint extension.
    :param verbose: Whether to print loaded state file.
    :return: The model state, optimizer state, number of steps trained, as well as number of epochs
             trained.
    """
    if not base_dir.exists():
        raise ValueError('Model directory for checkpoint loading does not exist:\n{}'
                         .format(base_dir))

    paths = list(base_dir.glob('{}_*.{}'.format(base_filename, extension)))
    if not paths:
        raise ValueError('Model directory for checkpoint loading does not contain any '
                         'checkpoints:\n{}'.format(base_dir))

    paths.sort(key=lambda pth: int(pth.stem.split('_')[-1]))
    latest_path = paths[-1]
    if verbose:
        print('Loading state file:', latest_path)
    state = torch.load(latest_path)
    return state['model_state'], state['optimizer_state'], state['num_steps_trained'], \
        state['num_epochs_trained']
