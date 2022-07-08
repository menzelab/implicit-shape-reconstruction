import argparse
import shutil
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import yaml

from impl_recon.utils import io_utils


class TaskType(IntEnum):
    """Type of task trained."""
    AD = 0  # Auto-decoder with implicit functions
    RN = 1  # Convolution auto-encoder (ReconNet)


def get_model_config_path(path_config: Dict, model_name: str, config_pattern: str) -> Path:
    """Get the path to model config from a given model directory."""
    model_dir = path_config['model_basedir'] / model_name
    model_config_filepath = io_utils.find_single_file(model_dir, config_pattern)
    return model_config_filepath


def write_config(source_config_file: Path, target_dir: Path) -> None:
    """Copy the source config file into the target directory."""
    if not target_dir.exists():
        raise ValueError('Target directory for writing config does not exist:\n{}'
                         .format(target_dir))

    target_file = target_dir / source_config_file.name
    shutil.copy(str(source_config_file), str(target_file))


def read_yaml_config(config_file_path: Union[Path, str],
                     default_config_file_path: Optional[Union[Path, str]] = None) -> Dict:
    if isinstance(config_file_path, str):
        config_file_path = Path(config_file_path)
    if not config_file_path.exists():
        raise ValueError(f'YAML config file does not exist:\n{config_file_path}')
    if default_config_file_path is not None:
        if isinstance(default_config_file_path, str):
            default_config_file_path = Path(default_config_file_path)
        if not default_config_file_path.exists():
            raise ValueError(f'Default YAML config file does not exist:\n'
                             f'{default_config_file_path}')
        with open(default_config_file_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = {}
    with open(config_file_path, 'r') as f:
        params.update(yaml.safe_load(f))
    return params


def read_train_config(config_file_path: Union[Path, str],
                      default_config_file_path: Optional[Union[Path, str]] = None) -> Dict:
    params = read_yaml_config(config_file_path, default_config_file_path)
    if params['model_name'] == 'None':
        params['model_name'] = None
    params['task_type'] = TaskType(params['task_type'])
    return params


def read_path_config(config_file_path: Union[Path, str]) -> Dict:
    params = read_yaml_config(config_file_path, None)
    for key in params:
        if '/' in params[key]:
            params[key] = Path(params[key])
    return params


def parse_config_train() -> Tuple[Dict, Path]:
    """For training: read path and the model configuration from a local YAML file.

    :return: the concatenated parameters and path to model config file.
    """
    parser = argparse.ArgumentParser(description='Train a neural network.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path_file', type=str)
    parser.add_argument('-c', '--config_file', type=str)
    args = parser.parse_args()
    if args.path_file is None:
        raise ValueError('Path config must be provided via a command-line argument.')
    if args.config_file is None:
        raise ValueError('Training config must be provided via a command-line argument.')

    config = read_path_config(args.path_file)
    model_config_path = Path(args.config_file)
    config.update(read_train_config(model_config_path, 'train_config_default.yml'))

    return config, model_config_path


def parse_config_eval() -> Tuple[Dict, Path]:
    """Return dictionary with merged model and evaluation parameters, as well as path to evaluation
    config file.
    """
    parser = argparse.ArgumentParser(description='Evaluate a neural network.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path_file', type=str)
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-c', '--eval_config', type=str)
    args = parser.parse_args()
    if args.path_file is None or args.model_name is None or args.eval_config is None:
        raise ValueError('Path config, evaluation config and model name must be provided.')

    config = read_path_config(args.path_file)
    model_config_path = get_model_config_path(config, args.model_name, '*.yml')
    config.update(read_train_config(model_config_path, 'train_config_default.yml'))
    eval_config_path = Path(args.eval_config)
    config.update(read_yaml_config(eval_config_path, 'eval_config_default.yml'))

    # The model name in the config file MUST coincide with the directory name!
    if config['model_name'] != args.model_name:
        raise ValueError(f'Cannot load stored model "{args.model_name}": it contains a different '
                         f'model name in the config: "{config["model_name"]}".')

    return config, eval_config_path
