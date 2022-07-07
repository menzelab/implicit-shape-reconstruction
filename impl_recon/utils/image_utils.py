import torch
from torch.nn import functional

from impl_recon.utils import geometry_utils


def interpolate_volume(source_volume: torch.Tensor, source_spacing: torch.Tensor,
                       source_offset: torch.Tensor, target_shape: torch.Tensor,
                       target_spacing: torch.Tensor, target_offset: torch.Tensor,
                       interpolation_type: str) -> torch.Tensor:
    """Interpolate a given volume with it's spacing and offset at a target resolution, spacing and
    offset.
    """
    # Again, align_corners=False:
    target_volume_size = target_shape * target_spacing
    # Reverse coordinates only in the end
    target_coords = geometry_utils.generate_sampling_grid(target_shape, -1, 0.0, target_volume_size)
    if not torch.allclose(target_offset, target_spacing / 2):
        target_coords = target_coords + (target_offset - (target_spacing / 2))
    lr_volume_size = torch.tensor(source_volume.shape) * source_spacing
    lr_bbox_offset = source_offset - (source_spacing / 2)
    target_coords = (target_coords - lr_bbox_offset) / lr_volume_size
    target_coords = 2 * target_coords - 1
    target_coords = target_coords[..., [2, 1, 0]]
    # Add batch and channel dimentions
    labels_batch = source_volume.unsqueeze(0).unsqueeze(0)
    interpol = functional.grid_sample(labels_batch, target_coords, interpolation_type, 'border',
                                      False).squeeze(0).squeeze(0)
    interpol = (interpol >= 0.5).to(source_volume.dtype)
    return interpol
