import logging
import os
import shutil
from collections import defaultdict
from glob import glob
from typing import List, Union, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from matplotlib import patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from skimage.segmentation import mark_boundaries
from src.model.model_interface import (ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)
from torch import Tensor
from scipy.stats import multivariate_normal

log = logging.getLogger(__name__)


def clean_predictions(model_dir: str, keep_step: Union[int, List[int]]):
    if isinstance(keep_step, int):
        keep_step = [keep_step]
    pred_dir = os.path.join(model_dir, 'predictions')
    step_folders = glob(f'{pred_dir}/step_*')
    kept_folders = [f'step_{step:09d}' for step in keep_step]
    for step_folder in step_folders:
        if os.path.basename(step_folder) not in kept_folders:
            log.info(f'Removing predictions: {step_folder}')
            shutil.rmtree(step_folder)


def prepare_prediction_dir(model_dir: str, step: int, prefix: str = None):
    prefix = '' if prefix is None else f'{prefix}_'
    pred_dir = os.path.join(model_dir, 'predictions', f'{prefix}step_{step:09d}')
    os.makedirs(pred_dir, exist_ok=True)
    return pred_dir


def plot_and_save_img_bboxes(
    model_dir: str,
    class_names: list,
    images: Tensor,
    target_boxes: List[Tensor],
    predicted_boxes: List[Tensor],
    step: int,
    sample_ids: List[str],
    prefix: str = None
):
    """
    :param model_dir:
    :param class_names:
    :param images: (N x H x W x 3)
    :param target_boxes:
    :param predictions:
    :return:
    """
    class_names = class_names[:-1]
    step_dir = prepare_prediction_dir(model_dir, step=step, prefix=prefix)

    assert len(predicted_boxes) == len(target_boxes) == len(sample_ids) == len(images), \
        f'{len(predicted_boxes)}, {len(target_boxes)}, {len(sample_ids)}, {len(images)}'
    class_cmap = color_map_for_classes(class_names, cmap='hsv')

    for sample_id, img_i, target_boxes_i, pred_boxes_i in zip(
            sample_ids, images, target_boxes, predicted_boxes):
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_img_with_bounding_boxes(ax, class_names, class_cmap,
                                     img=img_i, target_list=target_boxes_i,
                                     pred_list=pred_boxes_i)
        pred_path = os.path.join(step_dir, f'pred_{sample_id:03d}.png')
        fig.tight_layout()
        fig.savefig(pred_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_img_with_bounding_boxes(ax, class_names, class_cmap,
                                     img=img_i, target_list=target_boxes_i,
                                     pred_list=pred_boxes_i, plot_gaussian=True)
        pred_path = os.path.join(step_dir, f'pred_{sample_id:03d}_gauss.png')
        fig.tight_layout()
        fig.savefig(pred_path)
        plt.close(fig)


def plot_img_with_bounding_boxes(ax, class_names: list, class_cmap, img,
                                 pred_list=None, target_list=None,
                                 show_classes=False, plot_gaussian=False, plot_gt=True, plot_pred=True):
    # ax.xaxis.tick_top()
    if torch.is_tensor(img):
        # Convert to numpy and denormalize
        img = img.cpu().permute(1, 2, 0).numpy()
        img = img * 4.8828125e-4
        img = img + 0.5
        np.clip(img, 0, 1, out=img)
    img_size = img.shape[:2]
    ax.imshow(img, cmap='gray')
    if pred_list is not None and plot_pred:
        if torch.is_tensor(pred_list):
            pred_list = pred_list.detach().cpu().numpy()
        for box_prediction in pred_list:
            draw_box(ax, box_prediction, class_names, class_cmap, is_gt=False, img_size=img_size, plot_gaussian=plot_gaussian)
    if target_list is not None and plot_gt:
        if torch.is_tensor(target_list):
            target_list = target_list.detach().cpu().numpy()
        for box_prediction in target_list:
            draw_box(ax, box_prediction, class_names, class_cmap, is_gt=True, img_size=img_size, plot_gaussian=plot_gaussian)
    handles = [Line2D([0], [0], label='Target', color='black', linestyle='dashed', alpha=0.8),
               Line2D([0], [0], label='Pred', color='black', alpha=0.8)]
    if show_classes:
        handles += [patches.Patch(
            color=to_rgba(class_cmap[i], alpha=0.3),
            label=name
        ) for i, name in enumerate(class_names)]
    # if plot_gt and plot_pred:
    #     ax.legend(handles=handles)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def draw_box(ax, box_prediction, class_names: list, class_cmap, is_gt: bool, img_size, plot_gaussian=False, lw: int = 3):
    if torch.is_tensor(box_prediction):
        box_prediction = box_prediction.detach().cpu().numpy()
    if is_gt:
        x, y, w, h, class_id = box_prediction
    else:
        x, y, w, h, class_id, confidence = box_prediction
    class_name = class_names[int(class_id.item())]
    color = class_cmap[int(class_id.item())]

    ax.add_patch(patches.Rectangle(
        (x, y), w, h,
        fill=not plot_gaussian,
        facecolor=to_rgba(color, alpha=0.3),
        edgecolor=to_rgba(color, alpha=1.0),
        linestyle='dashed' if is_gt else 'solid',
        lw=lw
    ))
    if not is_gt and plot_gaussian:
        draw_gaussian(ax, box_prediction, class_cmap, img_size)
    if is_gt:
        ax.annotate(class_name, xy=(x, y), xytext=(2, 2),
                    textcoords='offset points', ha='left', va='bottom',
                    fontsize=18, color='white',
                    bbox={"facecolor": to_rgba(color, alpha=1.0),
                          "alpha": 1.0,
                          'pad': 2,
                          'lw': lw,
                          'edgecolor': 'none'})
    else:
        # f'{class_name}: {confidence:.2f}'
        ax.annotate(class_name, xy=(x, y + h), xytext=(2, -2),
                    textcoords='offset points', ha='left', va='top',
                    fontsize=18, color='white',
                    bbox={"facecolor": to_rgba(color, alpha=1.0),
                          "alpha": 1.0,
                          'pad': 2,
                          'lw': lw,
                          'edgecolor': 'none'})


def draw_gaussian(ax, box_prediction, class_cmap, img_size):
    H, W = img_size
    x = np.linspace(0, W, W, endpoint=False)
    y = np.linspace(0, H, H, endpoint=False)
    X, Y = np.meshgrid(x, y)

    if torch.is_tensor(box_prediction):
        box_prediction = box_prediction.detach().cpu().numpy()
    mu_x, mu_y, sigma_x, sigma_y, class_id, confidence = box_prediction
    mu_x, mu_y = mu_x + sigma_x / 2, mu_y + sigma_y / 2
    sigma_x = sigma_x * 10
    sigma_y = sigma_y * 10
    rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pd = rv.pdf(pos)
    pd = pd / pd.max()

    color = class_cmap[int(class_id.item())]
    colors = [to_rgba(color, alpha=al) for al in list(np.linspace(0.0, 0.8, 20))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap_', colors, 20)

    ax.contourf(X, Y, pd, levels=np.linspace(0.01, 1.0, 20), cmap=cmap)


COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#e377c2', '#ff7f0e', '#17becf', '#8c564b']


def color_map_for_classes(class_names, cmap=None):
    if len(class_names) <= len(COLORS):
        color_list = COLORS[:len(class_names)]
    else:
        cmap = plt.cm.get_cmap(cmap)
        color_list = cmap(np.linspace(0, 1, len(class_names)))
    return color_list


def plot_bounding_box_pred(
    ax,
    model: ObjectDetectorModelInterface,
    img: Tensor,
    target_boxes: list,
    class_names: list,
    cmap=None
):
    class_cmap = color_map_for_classes(class_names, cmap)

    predictions: ObjectDetectorPrediction = model.inference(img)
    plot_img_with_bounding_boxes(ax, class_names, class_cmap,
                                 predictions.box_prediction_hard, target_boxes)


def wandb_prepare_bboxes(preds: Tensor, targets: Tensor, class_names: List[str]):
    """
    Convert bboxes to wandb format

    :param preds: Tensor with predicted bounding boxes
                  (K x x, y, w, h, class_id, confidence)
    :param targets: Tensor with target bounding boxes
                    (K x x, y, w, h, class_id)
    :param class_names: List to map class_id to class_name
    """
    bboxes = defaultdict(lambda: defaultdict(list))

    # Add predicted bboxes
    for pred in preds:
        x_min, y_min, w, h, class_id, confidence = [p.item() for p in pred]
        class_id = int(class_id)
        bboxes['predictions']['box_data'].append({
            'position': {
                'minX': x_min,
                'minY': y_min,
                'maxX': x_min + w,
                'maxY': y_min + h,
            },
            "domain": "pixel",
            'class_id': class_id,
            'box_caption': class_names[class_id],
            'scores': {'confidence in %': confidence * 100}
        })
        bboxes['predictions']['class_labels'] = {
            i: name for i, name in enumerate(class_names)
        }

    # Add target bboxes
    for target in targets:
        x_min, y_min, w, h, class_id = [t.item() for t in target]
        class_id = int(class_id)
        bboxes['targets']['box_data'].append({
            'position': {
                'minX': x_min,
                'minY': y_min,
                'maxX': x_min + w,
                'maxY': y_min + h,
            },
            "domain": "pixel",
            'class_id': class_id,
            'box_caption': class_names[class_id]
        })
        bboxes['targets']['class_labels'] = {
            i: name for i, name in enumerate(class_names)
        }

    return dict(bboxes)


def wandb_prepare_masks(
    class_names: List[str],
    seg_mask_from_rois: Optional[Tensor] = None,
    seg_mask_from_patches: Optional[Tensor] = None
):
    """
    :param seg_mask_from_rois: segmentation masks derived from ROIs.
        If not None, will be plotted as "predictions".
        (H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param seg_mask_from_patches: segmentation masks derived from patches.
        If not None, will be plotted as "patch_preds".
        (H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param class_names: Names of the classes (including the no-finding/bg class
                        name as the last class)
    """
    if seg_mask_from_rois is None and seg_mask_from_patches is None:
        return None
    class_labels = {i: name for i, name in enumerate(class_names)}
    results = {}
    if seg_mask_from_rois is not None:
        results['predictions'] = {
            'class_labels': class_labels,
            'mask_data': seg_mask_from_rois.numpy()
        }
    if seg_mask_from_patches is not None:
        results['patch_preds'] = {
            'class_labels': class_labels,
            'mask_data': seg_mask_from_patches.numpy()
        }

    return results


def prepare_wandb_bbox_images(
    images: List[Tensor],
    preds: List[Tensor],
    targets: List[Tensor],
    seg_masks_from_rois: Optional[Tensor],
    seg_masks_from_patches: Optional[Tensor],
    class_names: List[str]
):
    """
    Usage: wandb.log(prepare_wandb_bbox_images(
        images,
        preds,
        targets,
        seg_masks_from_rois,
        seg_masks_from_patches,
        class_names
    ))

    :param images: List of images to log (N x C x H x W)
    :param preds: List of Tensors with predicted bounding boxes
                  (N x K x x, y, w, h, class_id, confidence)
    :param targets: List of Tensors with target bounding boxes
                    (N x M x x, y, w, h, class_id)
    :param seg_mask_from_rois: segmentation masks derived from ROIs.
        If not None, will be plotted as "predictions".
        (N x H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param seg_mask_from_patches: segmentation masks derived from patches.
        If not None, will be plotted as "patch_preds".
        (N x H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param class_names: Names of the classes (including the no-finding/bg
                        class name as the last class)
    """
    assert len(images) == len(preds) == len(targets)
    if seg_masks_from_rois is None:
        seg_masks_from_rois = [None for _ in range(len(images))]
    if seg_masks_from_patches is None:
        seg_masks_from_patches = [None for _ in range(len(images))]

    masks = [wandb_prepare_masks(class_names, seg_mask_from_rois, seg_mask_from_patches)
             for seg_mask_from_rois, seg_mask_from_patches
             in zip(seg_masks_from_rois, seg_masks_from_patches)]
    return [
        wandb.Image(
            image,
            boxes=wandb_prepare_bboxes(pred, target, class_names),
            masks=mask
        ) for image, pred, target, mask in zip(images, preds, targets, masks)
    ]


def prepare_wandb_superpixels(
    images: List[Tensor],
    sp_indices: List[Tensor],
    seg_masks_from_superpixels: Optional[Tensor],
    class_names: List[str]
):
    """
    Usage: wandb.log(prepare_wandb_superpixels(images, superpixels.sp_indices))

    Prepares logging of superpixels.

    :param images: List of images to log (N x C x H x W)
    :param sp_indices: Superpixl indices for each patch (N x H x W)
    :param seg_masks_from_superpixels: segmentation masks derived from superpixels.
        If not None, will be plotted as "patch_preds".
        (N x H x W) with integer values corresponding to (0-based) indices of
        class_names. For no-finding/bg use the index len(class_names)-1.
    :param class_names: Names of the classes (including the no-finding/bg
                        class name as the last class)
    """
    if seg_masks_from_superpixels is None:
        seg_masks_from_superpixels = [None for _ in range(len(images))]
    masks = [wandb_prepare_masks(class_names, seg_mask_from_sp)
             for seg_mask_from_sp
             in seg_masks_from_superpixels]

    return [
        wandb.Image(mark_boundaries(
            img, sp, color=(1, 1, 0), mode='outer'), masks=mask
        )
        for img, sp, mask
        in zip(
            images.permute(0, 2, 3, 1).expand(-1, -1, -1, 3).numpy(),
            sp_indices.numpy(),
            masks
        )
    ]


def wandb_clean_local(run):
    """
    Remove old images and bbox data from disk

    :param run: wandb run
    """
    wandb_img_dir = os.path.join(run.dir, 'media/images')
    if os.path.isdir(wandb_img_dir):
        for f in glob(f'{wandb_img_dir}/*.png'):
            os.remove(f)
    wandb_mask_dir = os.path.join(run.dir, 'media/images/mask')
    if os.path.isdir(wandb_mask_dir):
        for f in glob(f'{wandb_mask_dir}/*.mask.png'):
            os.remove(f)
    wandb_bbox_dir = os.path.join(run.dir, 'media/metadata/boxes2D')
    if os.path.isdir(wandb_bbox_dir):
        for f in glob(f'{wandb_bbox_dir}/*.json'):
            os.remove(f)


def wandb_clean_remote(run):
    """
    Remove old images and bbox data from wandb server

    :param run: wandb run
    """
    try:
        remote_run = wandb.Api(timeout=100).run(f"{run.entity}/{run.project}/{run.id}")
        for img in remote_run.files():
            if img.name.startswith('media/images/') or img.name.startswith('media/images/mask/') or img.name.startswith('media/metadata/boxes2D/'):
                img.delete()
    except Exception as e:
        log.warn(f'Error when cleaning images {e}')


def prepare_cm_dir(model_dir: str, prefix: str = None):
    cm_dir = os.path.join(model_dir, '' if prefix is None else prefix)
    os.makedirs(cm_dir, exist_ok=True)
    return cm_dir


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_dir: str,
    prefix: str = None
):
    """
    This function plots the confusion matrix.

    :param cm: Confusion matrix
    :param class_names: List of class names
    :param model_dir: Model directory
    :param prefix: Prefix to build the correct dir
    """
    class_names = class_names[:-1]
    # Set font and size
    #np.save("/vol/ada_ssd/users/meissen/workspace/weakly_supervised/object-detect-global-labels/cm.npy", cm)
    matplotlib.rcParams['font.family'] = "Verdana"
    params = {'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7}
    matplotlib.rcParams.update(params)

    cm_dir = prepare_cm_dir(model_dir, prefix=prefix)
    _, ax = plt.subplots(figsize=(3.48, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, linewidth=.5, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, verticalalignment='top')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.yticks(rotation=45, verticalalignment='top')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    # plt.savefig(os.path.join(cm_dir, 'rodeo_confusion_matrix.png'), dpi=300)
    plt.savefig(os.path.join(cm_dir, 'rodeo_confusion_matrix.pdf'))
    plt.close()
