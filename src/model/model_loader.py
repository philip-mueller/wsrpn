import logging
import os
from typing import Any, Dict, Union, Optional, Tuple

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.model.model_interface import ModelConfig, ObjectDetectorModelInterface
from src.utils.utils import prepare_config, get_model_dir, get_run_dir

log = logging.getLogger(__name__)


MODEL_CLASSES = {}


def register_model_class(model_cls):
    config_cls = model_cls.CONFIG_CLS
    model_type = model_cls.__name__

    assert model_type not in MODEL_CLASSES, f'{model_type} already registered'
    MODEL_CLASSES[model_type] = model_cls
    cs = ConfigStore.instance()
    cs.store(name=model_type, group='model', node=config_cls)


def load_model_from_config(config: Union[Dict[str, Any], ModelConfig]) \
        -> ObjectDetectorModelInterface:
    config = OmegaConf.create(config)
    assert 'model_type' in config.keys(), \
        f'model_type not in config. Keys: {config.keys()}'
    model_type = config.model_type
    assert model_type in MODEL_CLASSES, \
        f'{model_type} is not registered as a model_type. Available: {MODEL_CLASSES.keys()}'
    model_cls = MODEL_CLASSES[model_type]
    assert model_cls.CONFIG_CLS is not None, \
        f'Specify a CONFIG_CLS for the class {model_cls}'
    config = prepare_config(config, model_cls.CONFIG_CLS, log)

    return model_cls(config)


def load_model_from_checkpoint(checkpoint_path: str, return_dict=False) \
        -> Union[ObjectDetectorModelInterface,
                 Tuple[ObjectDetectorModelInterface, dict]]:
    ckpt_dict = torch.load(checkpoint_path)
    assert 'state_dict' in ckpt_dict and 'config_dict' in ckpt_dict, \
           f'Invalid checkpoint dict. Keys: {ckpt_dict.keys()}'
    config_dict = ckpt_dict['config_dict']
    state_dict = ckpt_dict['state_dict']

    model = load_model_from_config(config_dict)
    log.info(f'Loading model from checkpoint: {checkpoint_path}')
    model.load_state_dict(state_dict)
    if return_dict:
        return model, ckpt_dict
    else:
        return model


def load_model_by_name(
    model_name: str,
    run_name: Optional[str] = None,
    step: Optional[int] = None,
    load_best=False,
    return_dict=False
) -> Union[ObjectDetectorModelInterface,
           Tuple[ObjectDetectorModelInterface, dict]]:
    model_dir = get_model_dir(model_name)
    run_dir = get_run_dir(model_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')

    if load_best:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    else:
        if step == -1:
            checkpoints = [ckpt for ckpt in os.listdir(checkpoint_dir)
                           if ckpt.endswith('.pth') and not ckpt.endswith('_best.pth')]
            checkpoint_path = max(
                [os.path.join(checkpoint_dir, d) for d in checkpoints],
                key=os.path.getmtime
            )
            print(f'Latest checkpoint: {checkpoint_path}')
        else:
            assert step is not None
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f'checkpoint_{step:09d}.pth')
    return load_model_from_checkpoint(checkpoint_path, return_dict=return_dict)


def save_training_checkpoint(model, optimizer, lr_scheduler, scaler, results,
                             best_results, config, step, is_best=False):
    saved_states = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'amp': scaler.state_dict(),
        'step': step,
        'results': results,
        'best_results': best_results,
        'train_config': OmegaConf.to_container(config.training),
        'dataset': OmegaConf.to_container(config.dataset)
    }

    # Save the current model
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{step:09d}.pth')
    model.save_model(checkpoint_path, **saved_states)

    # Save as best model
    if is_best:
        checkpoint_path = os.path.join('checkpoints', 'checkpoint_best.pth')
        model.save_model(checkpoint_path, **saved_states)

    # Remove the previous model
    if step > 0:
        prev_checkpoint_path = os.path.join(
            'checkpoints', f'checkpoint_{step - config.val_freq:09d}.pth')
        if os.path.exists(prev_checkpoint_path):
            os.remove(prev_checkpoint_path)
