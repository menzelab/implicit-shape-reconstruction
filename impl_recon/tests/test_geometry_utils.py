from typing import List

import pytest
import torch

from impl_recon.utils import geometry_utils

test_shapes = [
    ([5] * 3),
    ([8] * 2)
]


@pytest.mark.parametrize('tgt_shape', test_shapes)
def test_generate_sampling_grid(tgt_shape: List[int]):
    device = torch.device('cpu')
    batch_size = 4

    ndims = len(tgt_shape)
    image_size = torch.arange(1, ndims + 1)
    tgt_shape_tensor = torch.tensor(tgt_shape)
    coords = geometry_utils.generate_sampling_grid(tgt_shape_tensor, -1, 0.0, image_size, device,
                                                   batch_size, False)

    assert list(coords.shape) == [batch_size, *tgt_shape, ndims]
    assert torch.equal(coords[0], coords[-1])
    spacing = image_size / tgt_shape_tensor
    if ndims == 2:
        start = coords[0, 0, 0]
        diag = coords[0, 1, 1] - start
        diag2 = coords[0, -1, -1] - coords[0, -2, -2]
        assert torch.allclose(coords[0, :, 0, 0], coords[0, :, 1, 0])
        assert torch.allclose(coords[0, 0, :, 1], coords[0, 1, :, 1])
    elif ndims == 3:
        start = coords[0, 0, 0, 0]
        diag = coords[0, 1, 1, 1] - start
        diag2 = coords[0, -1, -1, -1] - coords[0, -2, -2, -2]
        assert torch.allclose(coords[0, :, 0, 0, 0], coords[0, :, 1, 1, 0])
        assert torch.allclose(coords[0, 0, :, 0, 1], coords[0, 1, :, 1, 1])
        assert torch.allclose(coords[0, 0, 0, :, 2], coords[0, 1, 1, :, 2])
    else:
        raise ValueError(f'Unsupported dimensionality {ndims}.')

    assert torch.allclose(start, spacing / 2)
    assert torch.allclose(diag, spacing)
    assert torch.allclose(diag2, spacing)
