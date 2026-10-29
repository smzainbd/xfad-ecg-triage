"""1D Grad-CAM for ECG CNNs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grad_cam_1d_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_classes: torch.Tensor | int,
    last_conv_layer_name: str,
    *,
    task: str | None = None,
    create_graph: bool = False,
) -> torch.Tensor:
    """Grad-CAM for a whole batch, shape (batch, length), normalised per sample.

    With ``create_graph=True`` the CAM stays differentiable with respect to the model
    parameters (including those that only influence it through the gradient weights),
    which is what saliency-matching distillation needs. Explanation-only callers should
    leave it False: double backprop roughly doubles memory and time.
    """
    features: torch.Tensor | None = None

    def hook(_module, _inp, out):
        nonlocal features
        features = out

    target_layer = dict(model.named_modules())[last_conv_layer_name]
    handle = target_layer.register_forward_hook(hook)

    try:
        with torch.enable_grad():
            if not create_graph:
                x = x.clone().detach().requires_grad_(True)
            out = model(x, task, return_features=True) if task is not None else model(x, return_features=True)
            logits = out[0] if isinstance(out, tuple) else out

            if isinstance(target_classes, int):
                target_classes = torch.full((logits.shape[0],), target_classes, device=logits.device)
            target_classes = target_classes.to(logits.device).long().view(-1)

            assert features is not None
            # Summing over the batch keeps per-sample gradients independent, because
            # sample i's score does not depend on sample j's features.
            score = logits.gather(1, target_classes.unsqueeze(1)).sum()
            grads = torch.autograd.grad(
                score, features, retain_graph=create_graph, create_graph=create_graph,
            )[0]
    finally:
        handle.remove()

    weights = grads.mean(dim=2, keepdim=True)
    cam = F.relu((weights * features).sum(dim=1))
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False).squeeze(1)
    # Clamp rather than add an epsilon: raw CAM peaks can be ~1e-5, where a fixed
    # epsilon would visibly shrink the normalised map and distort the alignment loss.
    return cam / cam.amax(dim=-1, keepdim=True).clamp_min(1e-12)


def grad_cam_1d(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    last_conv_layer_name: str,
    *,
    task: str | None = None,
) -> torch.Tensor:
    """Return CAM of shape (batch, length) normalised per sample."""
    return grad_cam_1d_batch(model, x, target_class, last_conv_layer_name, task=task)
