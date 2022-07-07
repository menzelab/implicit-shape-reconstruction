import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import optimizer as opt
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation, io_utils, nn_utils


def create_model(params: dict, image_size: Optional[torch.Tensor]) -> torch.nn.Module:
    task_type = params['task_type']

    net: torch.nn.Module
    if task_type == config_io.TaskType.AD:
        if image_size is None:
            raise ValueError('Image size is required for AD model creation.')
        latent_dim = params['latent_dim']
        op_num_layers = params['op_num_layers']
        op_coord_layers = params['op_coord_layers']
        net = implicits.AutoDecoder(latent_dim, len(image_size), image_size,
                                    op_num_layers, op_coord_layers)
    elif task_type == config_io.TaskType.RN:
        net = implicits.ReconNet()
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    return net


def create_loss() -> torch.nn.Module:
    return nn_utils.BCEWithDiceLoss('mean', 1.0)


def train_one_epoch(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
                    latents: torch.nn.Parameter, lat_reg_lambda: float,
                    optimizer: opt.Optimizer, criterion: torch.nn.Module, metric: torch.nn.Module,
                    device: torch.device, epoch: int, num_epochs_target: int,
                    global_step: torch.Tensor, log_epoch_count: int,
                    logger: Optional[SummaryWriter], verbose: bool):
    loss_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    lat_reg = None
    t0 = time.time()
    net.train()
    for batch in ds_loader:
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        if task_type == config_io.TaskType.AD:
            latents_batch = latents[batch['caseids']].to(device)
            coords = batch['coords'].to(device)
            labels_pred = net(latents_batch, coords)
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
        elif task_type == config_io.TaskType.RN:
            labels_lr = batch['labels_lr'].to(device)
            labels_pred = net(labels_lr)
        else:
            raise ValueError(f'Unknown task type {task_type}.')
        loss = criterion(labels_pred, labels)
        if lat_reg is not None and lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss += min(1.0, epoch / 100) * lat_reg_lambda * lat_reg
        loss.backward()
        optimizer.step()

        loss_running += loss.item()
        num_losses += 1
        # Metric returns the sum
        metric_running += metric(labels_pred, labels).item()
        num_metrics += batch['labels'].shape[0]
        global_step += 1

    if epoch % log_epoch_count == 0:
        loss_avg = loss_running / num_losses
        metric_avg = metric_running / num_metrics
        num_epochs_trained = epoch + 1

        if logger is not None:
            logger.add_scalar('loss', loss_avg, global_step=num_epochs_trained)
            logger.add_scalar('metric/train', metric_avg, global_step=num_epochs_trained)
            if lat_reg is not None:
                # Log average squared norm of latents
                logger.add_scalar('lat_norm2/train', lat_reg.item(), global_step=num_epochs_trained)

        if verbose:
            epoch_duration = time.time() - t0

            print(f'[{num_epochs_trained}/{num_epochs_target}] '
                  f'Avg loss: {loss_avg:.4f}; '
                  f'metric: {metric_avg:.3f}; '
                  f'global step nb. {global_step} '
                  f'({epoch_duration:.1f}s)')


def optimize_latents(net: implicits.AutoDecoder,
                     latents_batch: torch.Tensor, labels: torch.Tensor, coords: torch.Tensor,
                     lr: float, lat_reg_lambda: float, num_iters: int,
                     device: torch.device,
                     max_num_const_train_dsc: int,
                     verbose: bool) -> None:
    """Optimize latent vectors for a single example.
    max_num_const_train_dsc: if train dice doesn't change this number of times, stop training. -1
                             means never stop early.
    """
    criterion = create_loss().to(device)
    optimizer_val = torch.optim.Adam([latents_batch], lr=lr)

    eval_every_x_steps = 10
    print_every_x_evals = 10
    prev_train_dsc = 0.0
    num_const_train_dsc = 0

    net.eval()

    t0 = time.time()
    for i in range(num_iters):
        labels_pred = net(latents_batch, coords)
        loss = criterion(labels_pred, labels)
        if lat_reg_lambda > 0:
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
            # Gradually build up regularization for the first 100 iters (follows DeepSDF)
            loss += min(1.0, i / 100) * lat_reg_lambda * lat_reg
        optimizer_val.zero_grad()
        loss.backward()
        optimizer_val.step()
        if (i + 1) % eval_every_x_steps == 0:
            dsc = nn_utils.dice_coeff(torch.sigmoid(labels_pred), labels, 0.5).item()
            if verbose and round((i + 1) / eval_every_x_steps) % print_every_x_evals == 0:
                print(f'Step {i + 1:04d}/{num_iters:04d}: loss {loss.item():.4f} DSC {dsc:.3f} '
                      f'L2^2(z): {torch.mean(torch.sum(torch.square(latents_batch), dim=1)):.2f} '
                      f'({time.time() - t0:.1f}s)')
                t0 = time.time()
            if round(dsc, 3) == round(prev_train_dsc, 3):
                num_const_train_dsc += 1
            else:
                num_const_train_dsc = 0
            if num_const_train_dsc == max_num_const_train_dsc:
                print(f'Reached stopping critertion after {i + 1} steps. '
                      f'Optimization has converged.')
                break
            prev_train_dsc = dsc


def validate(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
             latents: torch.nn.Parameter, metric: torch.nn.Module, device: torch.device, epoch: int,
             logger: Optional[SummaryWriter], verbose: bool):
    metric_running = 0.0
    num_metrics = 0

    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            labels = batch['labels'].to(device)
            if task_type == config_io.TaskType.AD:
                latents_batch = latents[batch['caseids']].to(device)
                coords = batch['coords'].to(device)
                labels_pred = net(latents_batch, coords)
            elif task_type == config_io.TaskType.RN:
                labels_lr = batch['labels_lr'].to(device)
                labels_pred = net(labels_lr)
            else:
                raise ValueError(f'Unknown task type {task_type}.')

            # Metric returns the sum
            metric_running += metric(labels_pred, labels).item()
            num_metrics += batch['labels'].shape[0]

        metric_avg = metric_running / num_metrics

    if logger is not None:
        logger.add_scalar('metric/val', metric_avg, global_step=(epoch + 1))

    if verbose:
        t1 = time.time()
        val_duration = t1 - t0
        print(f'[val] metric {metric_avg:.3f} ({val_duration:.1f}s)')


def main():
    params, config_filepath = config_io.parse_config_train()
    model_basedir: Path = params['model_basedir']
    model_dir = model_basedir / params['model_name'] if params['model_name'] is not None else None
    task_type = params['task_type']
    learning_rate = params['learning_rate'] * params['batch_size_train']
    lat_reg_lambda = params['lat_reg_lambda']
    num_epochs_target = params['num_epochs']
    log_epoch_count = params['log_epoch_count']
    checkpoint_epoch_count = params['checkpoint_epoch_count']
    max_num_checkpoints = params['max_num_checkpoints']

    writer: Optional[SummaryWriter]
    checkpoint_writer: Optional[io_utils.RollingCheckpointWriter]
    if model_dir is not None:
        if model_dir.exists():
            raise ValueError('Model directory already exists. Exiting to prevent accidental '
                             f'overwriting.\n{model_dir}')
        model_dir.mkdir()
        # Write the parameters to the model folder
        config_io.write_config(config_filepath, model_dir)

        # Redirect stdout to file + stdout
        sys.stdout = io_utils.Logger(model_dir / 'log.txt', 'a')
        writer = SummaryWriter(log_dir=str(model_dir))
        checkpoint_writer = io_utils.RollingCheckpointWriter(model_dir, 'checkpoint',
                                                             max_num_checkpoints, 'pth')
    else:
        print('Warning: no model name provided; not writing anything to the file system.')
        writer = None
        checkpoint_writer = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('Warning: no GPU available; training on CPU.')

    ds_loader_train = data_generation.create_data_loader(params, data_generation.PhaseType.TRAIN,
                                                         True)
    ds_loader_val = data_generation.create_data_loader(params, data_generation.PhaseType.VAL, True)
    image_size = ds_loader_train.dataset.image_size \
        if isinstance(ds_loader_train.dataset, data_generation.ImplicitDataset) else None

    if not ds_loader_train:
        raise ValueError(f'Number of training examples is smaller than the batch size.')

    net = create_model(params, image_size)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        net = torch.nn.DataParallel(net)  # experimental
    net = net.to(device)
    print(net)

    if task_type == config_io.TaskType.AD:
        lr_lats = params['learning_rate_lat']
        latent_dim = params['latent_dim']
        num_examples_train = len(ds_loader_train.dataset)  # type: ignore[arg-type]
        # Initialization scaling follows DeepSDF
        latents_train = torch.nn.Parameter(
            torch.normal(0.0, 1 / math.sqrt(latent_dim), [num_examples_train, latent_dim],
                         device=device))
        optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': learning_rate},
            {'params': latents_train, 'lr': lr_lats}
        ])
    else:
        latents_train = torch.nn.Parameter(torch.empty(0))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    criterion = create_loss().to(device)
    metric = nn_utils.DiceLoss(0.5, 'sum', True).to(device)

    # This is a tensor so that it is mutable within other functions
    global_step = torch.tensor(0, dtype=torch.int64)
    num_epochs_trained = 0

    for epoch in range(num_epochs_trained, num_epochs_target):
        train_one_epoch(task_type, ds_loader_train, net, latents_train, lat_reg_lambda, optimizer,
                        criterion, metric, device, epoch, num_epochs_target, global_step,
                        log_epoch_count, writer, True)
        if epoch % log_epoch_count == 0:
            validate(task_type, ds_loader_val, net, latents_train, metric, device, epoch, writer,
                     True)

        if checkpoint_writer is not None and epoch % checkpoint_epoch_count == 0:
            checkpoint_writer.write_rolling_checkpoint(
                {'net': net.state_dict(), 'latents_train': latents_train},
                optimizer.state_dict(), int(global_step.item()), epoch + 1)

    if checkpoint_writer is not None:
        checkpoint_writer.write_rolling_checkpoint(
            {'net': net.state_dict(), 'latents_train': latents_train},
            optimizer.state_dict(), int(global_step.item()), num_epochs_target)


if __name__ == '__main__':
    main()
