import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from impl_recon import train
from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation, impl_utils, io_utils, patch_utils


def export_batch(batch: dict, labels_pred: np.ndarray, spacings: np.ndarray,
                 task_type: config_io.TaskType, target_dir: Path):
    batch_casenames = batch['casenames']

    is_task_ad = task_type == config_io.TaskType.AD
    is_task_rn = task_type == config_io.TaskType.RN
    if is_task_ad:
        # Create a sparse volume with hi-res spacing/offset
        spacings_lr = batch['spacings_hr']
        offsets_lr = batch['offsets_hr']
        # GT does not have channels for AD
        label_values: torch.Tensor = batch['labels'].flatten(start_dim=1)
        voxel_ids = (batch['coords'] - offsets_lr) / spacings_lr
        voxel_ids = torch.round(voxel_ids).to(torch.int64).flatten(start_dim=1, end_dim=3)
        labels_lr = torch.zeros_like(batch['labels_hr'])
        labels_lr[:, voxel_ids[:, :, 0], voxel_ids[:, :, 1], voxel_ids[:, :, 2]] = label_values
        labels_lr = labels_lr.numpy()
        spacings_lr = spacings_lr.numpy()
        # Compute the offset w.r.t. hi-res volume (which is supposed to start at 0.0^3)
        # For sparse volumes with hi-res spacing/offset, it's zero
        # offsets_lr = (batch['offsets'] - batch['offsets_hr']).numpy()
        offsets_lr = np.zeros_like(spacings_lr)
    elif is_task_rn:
        labels_lr = batch['labels_lr'].squeeze(1).numpy()
        spacings_lr = batch['spacings'].numpy()
        offsets_lr = None
    else:
        raise ValueError(f'Unknown task type {task_type}.')

    for i in range(labels_pred.shape[0]):
        casename = batch_casenames[i]
        spacing = spacings[i]
        target_file = target_dir / f'{casename}_pred.nii.gz'
        io_utils.save_nifti_file(labels_pred[i, 0].astype(np.uint8), np.diag([*spacing, 1]),
                                 target_file)
        label_lr = labels_lr[i]
        spacing_lr = spacings_lr[i]
        ijk_to_lps = np.diag([*spacing_lr, 1])
        if is_task_ad and offsets_lr is not None:
            ijk_to_lps[:3, 3] = offsets_lr[i]
        target_file_lr = target_dir / f'{casename}_lr.nii.gz'
        io_utils.save_nifti_file(label_lr.astype(np.uint8), ijk_to_lps, target_file_lr)


def main():
    params, eval_config_path = config_io.parse_config_eval()

    evaluate_predictions = params['evaluate_predictions']
    export_predictions = params['export_predictions']
    allow_overwriting = params['allow_overwriting']
    lat_reg_lambda = params['lat_reg_lambda']
    latent_num_iters = params['latent_num_iters']
    max_num_const_train_dsc = params['max_num_const_train_dsc']
    task_type = params['task_type']
    model_dir = params['model_basedir'] / params['model_name']
    target_basedir = params['output_basedir']
    target_dirname = params['model_name']
    sample_orthogonal_slices = params['sample_orthogonal_slices']

    if not evaluate_predictions and not export_predictions:
        raise ValueError('Neither evaluation nor export were requested.')

    params['crop_size'] = 0
    params['batch_size_val'] = 1
    latent_lr = 1e-2
    if task_type == config_io.TaskType.AD:
        if sample_orthogonal_slices:
            target_dirname += f'_{latent_num_iters}_eval_ortho'
        else:
            target_dirname += f'_{latent_num_iters}_eval_ax{params["slice_step_axis"]}' \
                              f'_x{params["slice_step_size"]}'
    elif task_type == config_io.TaskType.RN:
        target_dirname += f'_eval_ax{params["slice_step_axis"]}_x{params["slice_step_size"]}'
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    target_dir = target_basedir / target_dirname
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_dim = params['latent_dim']

    if export_predictions:
        if not target_basedir.exists():
            raise ValueError(f'Target base directory does not exist:\n{target_basedir}')
        if target_dir.exists():
            if not allow_overwriting:
                raise ValueError(f'Target directory exists and overwriting is forbidden:\n'
                                 f'{target_dir}')
            else:
                for filepath in target_dir.glob('*'):
                    filepath.unlink()
        else:
            target_dir.mkdir()

        print(f'Writing results to: {target_dir}')
        # Redirect stdout to file + stdout
        sys.stdout = io_utils.Logger(target_dir / 'log.txt', 'w')
        # Write eval config
        config_io.write_config(eval_config_path, target_dir)

    # Print some info
    print(f'Evaluating model \'{params["model_name"]}\'.')
    if 'sample_orthogonal_slices' in params and params['sample_orthogonal_slices']:
        print('Reconstructing from three orthogonal slices.')
    else:
        print(f'Reconstructing from slices with step size {params["slice_step_size"]} '
              f'along axis {params["slice_step_axis"]}.')

    ds_loader = data_generation.create_data_loader(params, data_generation.PhaseType.INF, True)
    # During inference we rely on image size stored in with the model (it's read later)
    net = train.create_model(params, torch.ones(3, dtype=torch.float32))
    checkpoint = io_utils.load_latest_checkpoint(model_dir, 'checkpoint', 'pth', True)
    model_state = checkpoint[0]
    if 'net' not in model_state or 'latents_train' not in model_state:
        raise ValueError('Incompatible model state in stored checkpoint.')
    net_state = model_state['net']

    if 'latents_train' in model_state and model_state['latents_train'].numel() != 0:
        mean_sq_length = torch.mean(torch.sum(torch.square(model_state['latents_train']), dim=1))
        print(f'Train latents avg length squared: {mean_sq_length}')

    net.load_state_dict(net_state, strict=True)
    net = net.to(device)
    net.eval()

    # For AD, check that all volumes lie inside the training image_size
    if task_type == config_io.TaskType.AD:
        assert isinstance(ds_loader.dataset, data_generation.ImplicitDataset) or \
               isinstance(ds_loader.dataset, data_generation.OrthogonalSlices)
        assert isinstance(net, implicits.AutoDecoder)
        image_size_train = net.image_size.detach().cpu()
        image_size_curr = ds_loader.dataset.image_size
        # Allow some epsilon
        eps = 1e-2
        if torch.any(image_size_curr > image_size_train + eps):
            # This may not necessarily be a problem, but worth looking into if it happens
            raise ValueError(f'Max image size is larger than current model\'s: '
                             f'{image_size_curr} > {image_size_train} with epsilon {eps}.')

    all_dice_metrics: List[float] = []
    asds: List[float] = []
    hd95s: List[float] = []
    max_distances: List[float] = []

    for i, batch in enumerate(ds_loader):
        t0 = time.time()
        print(f'Batch {i + 1}/{len(ds_loader)}')
        labels_gt = batch['labels']
        # Target / GT spacings (are named differently for different tasks)
        spacings = batch['spacings'] if task_type != config_io.TaskType.AD else batch['spacings_hr']
        if task_type == config_io.TaskType.AD:
            labels_gt_sparse = batch['labels'].to(device)
            labels_gt = batch['labels_hr']  # no need to move to GPU
            if labels_gt_sparse.shape[0] != 1:
                raise ValueError(f'Only batch size 1 is supported for AD, instead got '
                                 f'{labels_gt_sparse.shape[0]}.')
            assert isinstance(net, implicits.AutoDecoder)
            print(f'Batch cases: {batch["casenames"]}')

            # Initialization scaling follows DeepSDF
            latents_batch = torch.nn.Parameter(
                torch.normal(0.0, 1e-4, [labels_gt_sparse.shape[0], latent_dim], device=device),
                requires_grad=True)

            coords = batch['coords'].to(device)
            train.optimize_latents(
                net, latents_batch, labels_gt_sparse, coords, latent_lr, lat_reg_lambda,
                latent_num_iters, device, max_num_const_train_dsc, True)
            print(f'L2^2(z): {torch.mean(torch.sum(torch.square(latents_batch), dim=1)):.2f}')
            # Full resolution prediction
            target_spatial_shape = torch.tensor(labels_gt.shape[1:])
            labels_pred = impl_utils.sample_latents(latents_batch, net,
                                                    target_spatial_shape, spacings)
            # Add channels for consistency with ReconNet
            labels_pred = labels_pred.unsqueeze(1)
            labels_gt = labels_gt.unsqueeze(1)
        elif task_type == config_io.TaskType.RN:
            labels_lr = batch['labels_lr'].to(device)
            with torch.no_grad():
                labels_pred = patch_utils.predict_tiled(labels_lr, net, params['rn_patch_size'])
        else:
            raise ValueError(f'Unknown task {task_type}.')
        print(f'Batch prediction time: {time.time() - t0:.1f}s')

        # Sigmoid + threshold
        labels_pred = torch.sigmoid(labels_pred).gt(0.5)
        labels_gt = labels_gt.to(torch.bool)

        labels_pred_cpu = labels_pred.cpu().numpy()
        labels_gt_cpu = labels_gt.cpu().numpy()
        spacings_cpu = spacings.numpy()

        if evaluate_predictions:
            impl_utils.eval_batch(labels_pred_cpu, labels_gt_cpu, spacings_cpu, all_dice_metrics,
                                  asds, hd95s, max_distances, True)
        if export_predictions:
            export_batch(batch, labels_pred_cpu, spacings_cpu, task_type, target_dir)

    print('\n')

    if all_dice_metrics and asds and hd95s and max_distances:
        print(f'ASD: {np.mean(asds):.2f} +- {np.std(asds):.2f} in '
              f'[{np.min(asds):.2f}, {np.max(asds):.2f}]')
        print(f'HSD95: {np.mean(hd95s):.2f} +- {np.std(hd95s):.2f} in '
              f'[{np.min(hd95s):.2f}, {np.max(hd95s):.2f}]')
        print(f'HSD: {np.mean(max_distances):.2f} +- {np.std(max_distances):.2f} in '
              f'[{np.min(max_distances):.2f}, {np.max(max_distances):.2f}]')
        print(f'DSC: {np.mean(all_dice_metrics):.2f} +- {np.std(all_dice_metrics):.2f} in '
              f'[{np.min(all_dice_metrics):.2f}, {np.max(all_dice_metrics):.2f}]')


if __name__ == '__main__':
    main()
