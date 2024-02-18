import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import MISSING, OmegaConf
from timm.scheduler import CosineLRScheduler
from torch import autocast, optim
from torch.cuda.amp import GradScaler
from torchmetrics import MetricCollection
from tqdm import tqdm
from torch import nn

from src.data.datasets import DatasetConfig, build_dataloaders
from src.data.transforms import TransformConfig
from src.metrics.avg_meter import AvgDictMeter, AvgMeter
from src.metrics import build_metrics
from src.model.inference import filter_top1_box_per_class
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)
from src.model.model_loader import (load_model_by_name, load_model_from_config,
                                    save_training_checkpoint)
from src.plot.plot_utils import (clean_predictions,
                                 prepare_wandb_bbox_images, wandb_clean_local,
                                 wandb_clean_remote)
from src.settings import (MODELS_DIR, PROJECT_DIR, WANDB_ENTITY,
                          WANDB_PROJECT)
from src.utils.utils import seed_everything, to_device
from src.evaluate import EvaluationConfig, evaluate

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    max_steps: int = MISSING
    lr: float = MISSING
    min_lr: float = MISSING
    warmup_lr: Optional[float] = MISSING
    warmup_steps: int = MISSING
    weight_decay: float = MISSING
    accumulation_steps: int = MISSING
    grad_clip_norm: Optional[float] = MISSING

    metric: str = MISSING
    metric_mode: str = MISSING
    seed: int = MISSING
    early_sopping_patience: Optional[int] = MISSING

    mixed_precision: bool = MISSING

    transform: TransformConfig = MISSING

    num_workers: int = MISSING
    prefetch: bool = MISSING

    top1_box_per_class: bool = MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING

    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING

    device: str = MISSING
    print_freq: int = MISSING
    val_freq: int = MISSING
    log_imgs_to_wandb: bool = MISSING
    debug: bool = MISSING
    resume: bool = MISSING
    run_name: Optional[str] = MISSING

    plot_predictions: int = MISSING
    keep_step_plots: bool = MISSING

    wandb_run_path: Optional[str] = None

    evaluate: bool = MISSING


def build_optimizer(model: nn.Module, train_config: TrainingConfig):
    return optim.AdamW(model.parameters(), lr=train_config.lr,
                       weight_decay=train_config.weight_decay)


def build_scheduler(optimizer, train_config: TrainingConfig):
    num_steps = int(train_config.max_steps)
    warmup_steps = int(train_config.warmup_steps)

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps, 
        lr_min=train_config.min_lr,
        warmup_lr_init=train_config.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )


def get_best_results(results, best_results, train_config: TrainingConfig):
    if best_results is None:
        return results, True
    assert train_config.metric_mode in ('min', 'max')
    best_value = best_results['val_metric']
    value = results['val_metric']
    if (value > best_value and train_config.metric_mode == 'max') or \
            (value < best_value and train_config.metric_mode == 'min'):
        return results, True
    else:
        return best_results, False


def train(config: ExperimentConfig):
    override_dirname = HydraConfig.get().job.override_dirname
    full_model_name = f'{config.name}/{override_dirname}' if len(override_dirname) > 0 else config.name
    if config.debug:
        log.info('Running in debug mode -> fast run to check for runtime errors')
        model_dir = None
        config.training.prefetch = False
        config.print_freq = 1
        config.val_freq = 1
    else:
        model_dir = HydraConfig.get().run.dir
        os.chdir(model_dir)

    seed_everything(config.training.seed)

    """ Load data """
    log.info(f"Loading data from {config.dataset.data_dir}")
    dataloaders, class_names = build_dataloaders(
        config.dataset,
        prefetch=config.training.prefetch,
        num_workers=config.training.num_workers,
        batch_size=config.training.batch_size,
        transform=config.training.transform,
        pixel_mean=config.dataset.pixel_mean,
        pixel_std=config.dataset.pixel_std,
        seed=config.training.seed,
    )
    config.model.num_classes = len(class_names)
    train_dataloader, val_dataloader = dataloaders['train'], dataloaders['val']

    """ Load model """
    if config.resume:
        model, checkpoint_dict = load_model_by_name(full_model_name, step=-1,
                                                    return_dict=True)
        start_step = checkpoint_dict['step'] + 1
        best_results = checkpoint_dict['best_results']
        log.info(f'Resuming training of {full_model_name} '
                 f'({type(model).__name__}) at step {start_step}')
    else:
        model: ObjectDetectorModelInterface = load_model_from_config(config.model)
        checkpoint_dict = None
        start_step = 0
        best_results = None
        log.info(f'Starting training of {full_model_name} ({type(model).__name__})')

    log.info(f'Model and training logs will be stored at: {model_dir}')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    """ Init W&B """
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=config.name,
        tags=[type(model).__name__],
        dir=model_dir,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        resume='must' if config.resume else False,
        reinit=True,  # without reinit there may be problems when running Hydra sweeps
        settings=wandb.Settings(start_method='thread', tmp_dir='~/tmp'),
        mode="disabled" if config.debug else "online" #"online",  # Easiest way to disable wandb
    )
    wandb.define_metric('img_step')
    wandb.define_metric('validation images', step_metric='img_step')
    wandb.summary['n_parameters'] = n_parameters
    log.info(f'Number of params: {n_parameters}')
    wandb_api_path = wandb.run.path
    config.wandb_run_path = wandb_api_path

    """ Save configs """
    if model_dir is not None:
        if config.resume:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            OmegaConf.save(
                config=config,
                f=os.path.join(model_dir, f'experiment_config__{timestamp}.yaml')
            )
        else:
            OmegaConf.save(config=config,
                           f=os.path.join(model_dir, 'experiment_config.yaml'))
            OmegaConf.save(config=model.config,
                           f=os.path.join(model_dir, 'model_config.yaml'))

    """ Build optimizer """
    optimizer = build_optimizer(model, config.training)
    lr_scheduler = build_scheduler(optimizer, config.training)
    scaler = GradScaler()
    if checkpoint_dict is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        scaler.load_state_dict(checkpoint_dict['amp'])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(config.device)

    # Exclude no finding label (last label) from evaluation metrics
    (detection_metrics,
     classification_metrics,
     auroc_metrics,
     accuracy_meter,
     box_metrics) = build_metrics(class_names[:-1])
    model = model.to(config.device)
    log.info(f"Using {config.device}")

    """ Start training """
    log.info(f'Starting training of {full_model_name}')
    step = 0
    avg_losses = AvgDictMeter()
    avg_loss = AvgMeter()
    pbar = tqdm(total=config.val_freq, desc='Training')

    while True:
        log.info(f'Starting epoch ({len(train_dataloader)} steps)')
        for samples in train_dataloader:
            pbar.update(1)

            # Training step
            loss, losses = train_step(model, samples, optimizer, lr_scheduler,
                                      scaler, step, config)
            avg_losses.add(losses)
            avg_loss.add(loss.detach())
            if torch.isnan(loss):
                log.error('Loss was nan')
                # scaler will ignore nan loss updated and we can therefore continue training

            # Increment step
            step += 1

            # Update progress bar and log losses
            if step % config.print_freq == 0 and not step % config.val_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': avg_loss.compute()})

                wandb.log({'train_step/lr': lr,
                           'train_step/loss': loss.detach(),
                           **{'train_step/' + key: value for key, value in losses.items()}},
                          step=step)

            # Validate and log at validation frequency
            if step % config.val_freq == 0:
                # Gather training results
                train_results = {
                    'loss': avg_loss.compute(),
                    **avg_losses.compute()
                }
                avg_loss = AvgMeter()
                avg_losses = AvgDictMeter()

                # Validate
                val_results = validate(
                    model,
                    val_dataloader,
                    detection_metrics,
                    classification_metrics,
                    auroc_metrics,
                    box_metrics,
                    accuracy_meter,
                    config,
                    model_dir,
                    class_names,
                    step=step
                )

                results = {
                    'step': step,
                    **{'train/' + key: value for key, value in train_results.items()},
                    **{'val/' + key: value for key, value in val_results.items()},
                    'val_metric': val_results[config.training.metric],
                    'val/predictions': wandb.plot.bar(
                        wandb.Table(
                            data=[[n, c] for n, c in zip(class_names, val_results['pred_counts'])],
                            columns=['class', 'count']
                        ),
                        'class',
                        'count',
                        title='Predicted class distribution'
                    )
                }

                if config.debug:
                    log.info('Debug mode -> Stopping training after 1 step')
                    return wandb_api_path, results

                best_results, is_best = get_best_results(results, best_results,
                                                         config.training)
                best_results = dict(best_results)
                if 'val/predictions' in best_results:
                    best_results.pop('val/predictions')
                save_training_checkpoint(model, optimizer, lr_scheduler, scaler,
                                         results=results,
                                         best_results=best_results,
                                         config=config, step=step,
                                         is_best=is_best)
                wandb.log(results, step=step)
                wandb.run.summary.update(best_results)

                # Cleanup prediction plots
                if config.plot_predictions > 0 and not config.keep_step_plots:
                    clean_predictions(model_dir, keep_step=best_results['step'])

                if is_best:
                    log.info(f'Step {step} (val/{config.training.metric}='
                             f'{results["val_metric"]}) -> best step')
                else:
                    best_step_diff = step - best_results['step']
                    log.info(f'Step {step} (val/{config.training.metric}='
                             f'{results["val_metric"]}) '
                             f'-> outperformed by step {best_results["step"]} '
                             f'(val/{config.training.metric}='
                             f'{best_results["val_metric"]})')

                    if config.training.early_sopping_patience is not None:
                        if best_step_diff > config.training.early_sopping_patience:
                            log.info(f'Early stopping: '
                                     f'val/{config.training.metric} did not'
                                     f'improve for {best_step_diff} steps '
                                     f'-> stopping training')
                            if best_results is not None:
                                log.info(f'Best step: {best_results["step"]} '
                                         f'(val/{config.training.metric}='
                                         f'{best_results["val_metric"]})')
                            wandb.finish()
                            return wandb_api_path, best_results
                        else:
                            log.info(f'Early stopping: '
                                     f'val/{config.training.metric} did not'
                                     f'improve for {best_step_diff} steps - '
                                     f'patience={config.training.early_sopping_patience}')

                # Reset progress bar
                pbar.refresh()
                pbar.reset()

            # Return if max_steps is reached
            if step >= config.training.max_steps:
                log.info(f'Finished training after {results["step"]} steps')
                if best_results is not None:
                    log.info(f'Best step: {best_results["step"]} '
                             f'(val/{config.training.metric}='
                             f'{best_results["val_metric"]})')
                wandb.finish()
                return wandb_api_path, best_results


def train_step(model, samples, optimizer, lr_scheduler, scaler, step, config):
    # Forward
    x, target_label, bboxes = to_device(samples, config.device)
    with autocast(device_type=config.device, enabled=config.training.mixed_precision):
        loss, losses, _ = model.train_step(x, target_label,
                                           target_boxes=bboxes,
                                           step=step)
        loss = loss / config.training.accumulation_steps

    # Backward and scale if mixed precision
    scaler.scale(loss).backward()

    # Update
    if (step + 1) % config.training.accumulation_steps == 0:
        if config.training.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step_update(step)

    return loss, losses


def validate(
    model,
    val_dataloader,
    detection_metrics,
    classification_metrics: MetricCollection,
    auroc_metrics,
    box_metrics,
    accuracy_meter,
    config: ExperimentConfig,
    model_dir,
    class_names,
    step
):
    model.eval()

    # Reset metrics
    detection_metrics.reset()
    box_metrics.reset()
    classification_metrics.reset()
    for metric in auroc_metrics.values():
        metric.reset()
    avg_losses = AvgDictMeter()
    avg_loss = AvgMeter()
    pred_counts = torch.zeros(len(class_names), dtype=torch.int64)

    if config.log_imgs_to_wandb:
        # Remove old images and bounding boxes (from previous epoch)
        wandb_clean_local(wandb.run)
        wandb_clean_remote(wandb.run)

    # Init progress bar
    pbar = tqdm(val_dataloader)
    pbar.set_description(f'Validate at step {step}')

    # Iterate over batches
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
        with torch.no_grad():
            with autocast(device_type=config.device, enabled=False):
                predictions: ObjectDetectorPrediction
                loss, losses, predictions = model.train_step(
                    x, target_label,
                    target_boxes=target_boxes,
                    return_predictions=True,
                    step=step
                )
            avg_loss.add(loss.detach())
            avg_losses.add(losses)

        # Cut no finding from target_label if necessary
        target_label_cpu = target_label_cpu[
            :, :predictions.global_prediction_hard.shape[1]]

        # If activated, only consider most confident box per class
        if config.training.top1_box_per_class:
            predictions = filter_top1_box_per_class(predictions)

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
        detection_metrics.add(predictions.box_prediction_hard, target_boxes_cpu)

        # For logging predicted classes
        pred_counts += torch.cat(
            [predictions.global_prediction_probs,
             predictions.global_obj_probs[:, None]],
            dim=1
        ).argmax(dim=1).bincount(minlength=len(class_names))

        # Plotting
        first_sample_index = idx * config.training.batch_size
        if first_sample_index < config.plot_predictions and not config.debug:
            # Seg masks now back to CPU
            seg_masks_from_patches_cpu = to_device(
                predictions.seg_masks_from_patches, "cpu")
            seg_masks_from_rois_cpu = to_device(
                predictions.seg_masks_from_rois, "cpu")

            # Only plot subset of current batch
            sample_index_end = (idx + 1) * config.training.batch_size
            if sample_index_end > config.plot_predictions:
                sample_index_end = config.plot_predictions
                x_cpu = x_cpu[:(sample_index_end - first_sample_index)]
                target_boxes_cpu = target_boxes_cpu[
                    :(sample_index_end - first_sample_index)]
                predictions.box_prediction_hard = predictions.box_prediction_hard[
                    :(sample_index_end - first_sample_index)]
                if seg_masks_from_patches_cpu is not None:
                    seg_masks_from_patches_cpu = seg_masks_from_patches_cpu[
                        :(sample_index_end - first_sample_index)]
                if seg_masks_from_rois_cpu is not None:
                    seg_masks_from_rois_cpu = seg_masks_from_rois_cpu[
                        :(sample_index_end - first_sample_index)]
            # Log images to wandb
            if config.log_imgs_to_wandb:
                # Log new images
                wandb.log({
                    'validation images': prepare_wandb_bbox_images(
                        x_cpu, predictions.box_prediction_hard,
                        target_boxes_cpu, seg_masks_from_rois_cpu,
                        seg_masks_from_patches_cpu, class_names
                    ),
                }, step=step)

        if config.debug:
            break  # single iteration

    # Returns
    val_results = {
        'loss': avg_loss.compute(),
        **avg_losses.compute(),
        **{'acc2/' + k if k != 'overall' else 'acc2': v for k, v in accuracy_meter.compute().items()},
        **detection_metrics.compute(),
        **classification_metrics.compute(),
        **box_metrics.compute(),
        'pred_counts': pred_counts,
        'auroc': auroc_metrics['global'].compute()
    }
    if len(auroc_metrics['patch_aggregated'].preds) > 0:
        val_results['auroc_patch'] = auroc_metrics['patch_aggregated'].compute()
    if len(auroc_metrics['roi_aggregated'].preds) > 0:
        val_results['auroc_roi'] = auroc_metrics['roi_aggregated'].compute()

    model.train()
    return val_results


def update_summary_with_api(results: Dict, wandb_run_api_path: str) -> None:
    if not isinstance(wandb_run_api_path, wandb.sdk.lib.disabled.RunDisabled):
        wandb_run_api = wandb.Api(timeout=120).run(wandb_run_api_path)
        for key, value in results.items():
            wandb_run_api.summary[key] = value
        wandb_run_api.update()


def build_eval_config(config: ExperimentConfig) -> EvaluationConfig:
    override_dirname = HydraConfig.get().job.override_dirname
    model_name = os.path.join(config.name, override_dirname)
    return EvaluationConfig(
        model_name=model_name,
        run_name=os.path.join(*HydraConfig.get().run.dir.split('/')),
        eval_prefix=f'{config.dataset.name}_test_{override_dirname}',
        dataset=config.dataset,
        dataset_type='test',
        device=config.device,
        num_workers=config.training.num_workers,
        prefetch=config.training.prefetch,
        seed=config.training.seed,
        debug=config.debug
    )


@hydra.main(config_path="conf", config_name="main_config")
def run_training(config):
    wandb_run_api_path, results = train(config)
    update_summary_with_api(results, wandb_run_api_path)
    if config.evaluate and not config.debug:
        eval_config = build_eval_config(config)
        evaluate(eval_config)
    return results['val_metric']


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=ExperimentConfig)
    OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver("ifel", lambda flag, val_true, val_false: val_true if flag else val_false)
    OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR)
    run_training()
