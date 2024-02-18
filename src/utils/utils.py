import glob
import logging
import os
import random
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import wandb
from deepdiff import DeepDiff
from omegaconf import OmegaConf
from wandb.apis.public import Run

from src.settings import WANDB_ENTITY, WANDB_PROJECT, MODELS_DIR
log = logging.getLogger(__name__)


def prepare_config(config, config_cls, log):
    # make it possible to init this class with different types of configs (dataclass, omegaconf, dict)
    config = OmegaConf.create(config)
    # fill defaults, which is required if "deprecated" configs are used (e.g. when loading old checkpoints)
    config_defaults = OmegaConf.structured(config_cls)
    new_config = OmegaConf.merge(config_defaults, config)
    diff = DeepDiff(config, new_config, verbose_level=2)
    if len(diff) > 0:
        log.info(f'Defaults have been added to the config: {diff}')
    return new_config


def config_to_dict(config):
    return OmegaConf.to_container(OmegaConf.create(config))


def to_device(data, device: str):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        non_blocking = device != 'cpu'
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [to_device(d, device) for d in data]
    else:
        raise TypeError(type(data))


def get_wandb_run_from_id(run_id) -> Run:
    return wandb.Api().run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")


def get_wandb_run_from_model_name(
    model_name: str,
    run_name: Optional[str] = None
) -> Run:
    model_dir = get_model_dir(model_name)
    run_dir = get_run_dir(model_dir, run_name)
    wandb_dir = os.path.join(run_dir, 'wandb')
    wandb_files = glob.glob(f'{wandb_dir}/run-*')
    if len(wandb_files) != 1:
        raise AssertionError(f'Multiple or no wandb runs found: '
                             f'{wandb_files}\nDir: {wandb_dir}')
    run_file = wandb_files[0]
    run_id = run_file[-8:]
    return get_wandb_run_from_id(run_id)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model_dir(name: str) -> str:
    model_dir = os.path.join(MODELS_DIR, name)
    assert os.path.exists(model_dir), f'Model {name} does not exist: {model_dir}'
    model_dir_subfolders = [
        f.name
        for f in os.scandir(model_dir)
        if f.is_dir() and not f.name.startswith('.') and not f.name.startswith('eval_')]
    assert len(model_dir_subfolders) > 0, (f'Model folder of model {name} is '
                                           f'empty: {model_dir}')

    if any(f.startswith('run_') for f in model_dir_subfolders):
        return model_dir
    elif len(model_dir_subfolders) == 1:
        submodel = f'{name}/{model_dir_subfolders[0]}'
        log.info(f'Found single submodel {submodel}. Using this model')
        return get_model_dir(submodel)
    else:
        raise AssertionError(f'Model folder of model {name} ({model_dir})'
                             f' contains multiple submodels but no runs:'
                             f' {model_dir_subfolders}')


def get_run_dir(model_dir: str, run_name: Optional[str] = None) -> str:
    runs = [d[0] for d in os.walk(model_dir) if d[0].split('/')[-1].startswith('run_')]
    if run_name is not None:
        try:
            run = [r for r in runs if run_name in r][0]
        except IndexError:
            raise ValueError(f'Run {run_name} not found in {model_dir}')
    else:
        runs = sorted(runs)
        run = runs[-1]
    return os.path.join(model_dir, run)


def create_model_dir(name: str, resume) -> str:
    model_dir = os.path.join(MODELS_DIR, name)
    if resume:
        assert os.path.exists(model_dir), (f'Cannot resume, model {name} does'
                                           f' not exist: {model_dir}')
    else:
        assert not os.path.exists(os.path.join(model_dir, 'wandb')), \
            f'Model {name} already exists: {model_dir}'
        os.makedirs(model_dir, exist_ok=True)
    return model_dir
