"""Two complementary attribution methods.

Integrated Gradients (IG): lead x time map. Baseline = all zeros in the
normalised input, i.e. every lead flat at its training-set mean ("no ECG").

Grad-CAM: time-only map from the last convolutional stage, stretched back
to the full 5,000 samples. It has no lead resolution, so lead-level
results come from IG only.
"""
import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam

from src import config


class Explainer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.eval()
        self.device = device
        self.ig = IntegratedGradients(self.model)
        self.gc = LayerGradCam(self.model, self.model.last_conv)

    def integrated_gradients(self, x: torch.Tensor) -> np.ndarray:
        """x: [batch, 12, samples] normalised. Returns [batch, 12, samples]."""
        x = x.to(self.device).clone().requires_grad_(True)   # a copy, so the caller's tensor is untouched
        attr = self.ig.attribute(x, baselines=torch.zeros_like(x),
                                 n_steps=config.XAI_IG_STEPS, internal_batch_size=64)
        return attr.detach().float().cpu().numpy()

    def grad_cam(self, x: torch.Tensor) -> np.ndarray:
        """x: [batch, 12, samples]. Returns [batch, samples] (time only)."""
        x = x.to(self.device).clone().requires_grad_(True)   # a copy, so the caller's tensor is untouched
        cam = self.gc.attribute(x, relu_attributions=True)          # [batch, 1, T']
        cam = LayerAttribution.interpolate(cam, (x.shape[-1],), interpolate_mode="linear")
        return cam.detach().float().cpu().numpy()[:, 0, :]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation between two maps (no scipy dependency on ties needed)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")
