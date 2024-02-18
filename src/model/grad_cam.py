"""Adapted from https://github.com/jacobgil/pytorch-grad-cam"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_2d_projection(acts: Tensor) -> Tensor:
    """Get a projection of the activations with SVD
    :param acts: (N, d, h, w) batch of activations from target layer
    """
    N, d, h, w = acts.shape
    acts[torch.isnan(acts)] = 0
    projections = []
    for act in acts:  # (d, h, w)
        reshaped_act = act.reshape(d, -1).T  # (h*w, d)
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_act -= reshaped_act.mean(0, keepdim=True)  # (h*w, d)
        U, S, Vh = torch.linalg.svd(reshaped_act)
        # U: (h*w, h*w), S: (h*w), Vh: (d, d)
        projection = reshaped_act @ Vh[0]  # (h*w)
        projection = projection.reshape(h, w)  # (h, w)
        projections.append(projection)
    return torch.stack(projections, dim=0)[:, None]  # (N, 1, h, w)


class GradCAM:
    def __init__(
        self,
        target_layer: nn.Module,
    ) -> None:
        self.gradients = None
        self.activations = None
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.activation_hook))
        self.handles.append(
            target_layer.register_forward_hook(self.gradient_hook))

    def activation_hook(self, module, input, output):
        self.activations = output.detach()

    def gradient_hook(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        def _store_grad(grad):
            self.gradients = grad.detach()

        output.register_hook(_store_grad)

    def get_activations(self):
        assert self.activations is not None, "Run forward first"
        acts = self.activations
        self.activations = None
        return acts

    def get_gradients(self):
        assert self.gradients is not None, "Compute loss and .backward() first"
        grads = self.gradients
        self.gradients = None
        return grads

    def __call__(self, x: Tensor, eig_smooth: bool = False) -> Tensor:
        """Perform GradCAM. Get activations and gradients of target layer
        and generate a heatmap.

        :param x: (N, C, H, W) Input images
        """
        # Get activations and gradients of target layer
        acts = self.get_activations()  # (N, d, h, w)
        grads = self.get_gradients()  # (N, d, h, w)

        # Actual GradCAM
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (N, d, 1, 1)
        cam = acts * weights  # (N, d, h, w)

        if eig_smooth:
            if cam.device != 'cpu':
                raise NotImplementedError("Not on cuda implemented yet")
            cam = get_2d_projection(cam)  # (N, 1, h, w)
        else:
            cam = cam.mean(1, keepdim=True)  # (N, 1, h, w)

        # Normalize heatmap
        cam_max = cam.amax((1, 2, 3), keepdim=True)  # (N, 1, 1, 1)
        cam_min = cam.amin((1, 2, 3), keepdim=True)  # (N, 1, 1, 1)
        cam = (cam - cam_min) / (1e-7 + cam_max)  # (N, 1, h, w)

        # Resize
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear',
                            align_corners=False)  # (N, 1, H, W)

        # Omit channel dimension (is 1)
        cam = cam[:, 0]  # (N, H, W)

        return cam

    def release(self):
        for handle in self.handles:
            handle.remove()
