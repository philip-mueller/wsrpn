import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

from src.data.datasets import build_dataloaders
from src.metrics import build_metrics
from src.model.model_loader import load_model_by_name
from src.plot.plot_utils import plot_and_save_img_bboxes, plot_confusion_matrix
from src.settings import MODELS_DIR
from src.utils.utils import (get_model_dir, get_run_dir, get_wandb_run_from_model_name, seed_everything,
                             to_device)

log = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    model_name: str = MISSING
    run_name: Optional[str] = MISSING
    eval_prefix: str = MISSING

    dataset: Any = MISSING
    dataset_type: str = MISSING
    inference: dict = field(default_factory=dict)

    plot: bool = MISSING
    update_wandb: bool = MISSING
    bootstrap: bool = MISSING

    device: str = MISSING
    num_workers: int = MISSING
    prefetch: bool = MISSING
    seed: int = MISSING
    debug: bool = MISSING


def evaluate(config: EvaluationConfig):
    """"""""""""""""""""""""""""""" Setup """""""""""""""""""""""""""""""
    log.info(f'Starting Evaluation of {config.model_name} '
             f'({config.dataset.name} -> {config.dataset_type})')
    log.info(f'Inference arguments: {config.inference}')
    model_dir = get_run_dir(get_model_dir(config.model_name), config.run_name)
    model, checkpoint_dict = load_model_by_name(
        config.model_name,
        run_name=config.run_name,
        load_best=True,
        return_dict=True
    )
    model.eval()
    step = checkpoint_dict['step']
    log.info(f'Evaluating step {step}')

    wandb_run = None
    if not config.debug and config.update_wandb:
        try:
            wandb_run = get_wandb_run_from_model_name(
                config.model_name,
                run_name=config.run_name
            )
            assert wandb_run.state != 'running', 'Run is still running'
        except Exception as e:
            log.error(f'Could not get wandb run: {e}\n'
                      'Evaluating without saving to wandb.')
            wandb_run = None
    train_config = OmegaConf.create(checkpoint_dict['train_config'])
    dataset_config = OmegaConf.create(checkpoint_dict['dataset'])

    seed_everything(config.seed)

    # Load data
    dataloaders, class_names = build_dataloaders(
        config.dataset,
        prefetch=False,
        num_workers=int(train_config.num_workers),
        batch_size=train_config.batch_size,
        transform=train_config.transform,
        pixel_mean=dataset_config.pixel_mean,
        pixel_std=dataset_config.pixel_std,
        seed=config.seed,
    )
    eval_dataloader = dataloaders[config.dataset_type]

    (detection_metrics,
     classification_metrics,
     auroc_metrics,
     accuracy_meter,
     box_metrics) = build_metrics(class_names[:-1], bootstrap=config.bootstrap)

    model = model.to(config.device)
    torch.backends.cudnn.benchmark = True
    log.info(f'Using {config.device}')

    """"""""""""""""""""""""""""""" Evaluate """""""""""""""""""""""""""""""
    pbar = tqdm(eval_dataloader)
    pbar.set_description(f'Evaluate ({config.dataset_type})')
    with torch.no_grad():
        for idx, samples in enumerate(pbar):
            x, target_label, target_boxes = samples

            # Keep copies on CPU for metrics and logging
            x_cpu = x
            target_label_cpu = target_label
            target_boxes_cpu = target_boxes

            # To GPU
            x = to_device(x, config.device)
            target_label = to_device(target_label, config.device)
            target_boxes = to_device(target_boxes, config.device)

            predictions = model.inference(x)

            # Cut no finding from target_label if necessary
            target_label_cpu = target_label_cpu[
                :, :predictions.global_prediction_hard.shape[1]]

            # Metrics that are faster on CPU
            classification_metrics.update(
                predictions.global_prediction_hard,
                target_label_cpu
            )
            accuracy_meter.update(
                predictions.global_prediction_probs.float(),
                target_label_cpu
            )
            auroc_metrics['global'].update(
                predictions.global_prediction_probs,
                target_label_cpu
            )
            if predictions.aggregated_roi_prediction_probs is not None:
                auroc_metrics['roi_aggregated'].update(
                    predictions.aggregated_roi_prediction_probs,
                    target_label_cpu
                )
            if predictions.aggregated_patch_prediction_probs is not None:
                auroc_metrics['patch_aggregated'].update(
                    predictions.aggregated_patch_prediction_probs,
                    target_label_cpu
                )
            box_metrics.add(predictions.box_prediction_hard)
            detection_metrics.add(predictions.box_prediction_hard,
                                  target_boxes_cpu)

            # ----- plot -----
            if config.plot:
                bs = train_config.batch_size // train_config.transform.n_views
                first_sample_index = idx * bs
                plot_and_save_img_bboxes(
                    model_dir,
                    class_names,
                    x_cpu,
                    target_boxes_cpu,
                    predictions.box_prediction_hard,
                    step=step,
                    sample_ids=list(range(first_sample_index, first_sample_index + len(predictions.box_prediction_hard))),
                    prefix=f'eval_{config.eval_prefix}'
                )

    """"""""""""""""""""""""""""""" Log """""""""""""""""""""""""""""""

    detection_metrics_kwargs = {'csv_path': f'{model_dir}/eval_{config.eval_prefix}_detection.csv'} if config.bootstrap else {}
    results = {
        **{'acc2/' + k if k != 'overall' else 'acc2': v
           for k, v in accuracy_meter.compute().items()},
        **detection_metrics.compute(**detection_metrics_kwargs),
        **classification_metrics.compute(),
        **box_metrics.compute(),
        'auroc': auroc_metrics['global'].compute()
    }

    # plot_confusion_matrix(
    #     cm=results['RoDeO/confusion_matrix'],
    #     class_names=class_names[:-1],
    #     model_dir=model_dir,
    #     prefix=f'eval_{config.eval_prefix}'
    # )

    config_dict = {
        'dataset': OmegaConf.to_container(config.dataset),
        'dataset_type': config.dataset_type,
        **config.inference
    }

    log.info('Finished evaluating')
    log.info(f'Results: {pformat(results)}')
    if wandb_run is not None and config.update_wandb:
        wandb_run.config.update({f'eval_{config.eval_prefix}': config_dict})
        wandb_run.summary.update({f'eval_{config.eval_prefix}/{key}': val
                                  for key, val in results.items()})


@hydra.main(config_path="conf", config_name="eval_config")
def run_evaluate(config):
    evaluate(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="eval", node=EvaluationConfig)
    OmegaConf.register_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver(
        "ifel",
        lambda flag, val_true, val_false: val_true if flag else val_false
    )
    run_evaluate()
