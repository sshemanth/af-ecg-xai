"""Attribution methods.

Integrated Gradients (IG): lead x time map. Baseline = all zeros in the
normalised input, i.e. every lead flat at its training-set mean ("no ECG").

Grad-CAM: time-only map from the last convolutional stage, stretched back
to the full 5,000 samples. It has no lead resolution, so lead-level
results come from IG only.
"""
import copy

import numpy as np
import torch
from captum.attr import (DeepLift, GradientShap, InputXGradient, IntegratedGradients,
                         LayerAttribution, LayerGradCam, Occlusion)

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


# ---------------------------------------------------------------------------
# More methods, for the comparison step (scripts/11_attribution_methods.py)
# ---------------------------------------------------------------------------
def no_inplace(model: torch.nn.Module) -> torch.nn.Module:
    """A copy of the model with in-place ReLUs switched off; DeepLIFT's hooks
    need each activation as a separate tensor."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
    return model.eval()


class MultiExplainer:
    """Six attribution methods on the same model output (the AF logit).

    Lead x time maps:  Integrated Gradients, DeepLIFT, GradientSHAP,
                       Input x Gradient
    Time-only maps:    Grad-CAM (last conv stage), Occlusion (sliding
                       100 ms window across all leads, set to the training
                       mean; the drop in the logit is the attribution)
    Zero baseline = every lead flat at its training-set mean. GradientSHAP
    draws its baselines from a background set of training ECGs instead.
    """
    METHODS = ["IG", "DeepLIFT", "GradientSHAP", "InputXGradient", "Grad-CAM", "Occlusion"]

    def __init__(self, model: torch.nn.Module, device: torch.device, background: torch.Tensor,
                 occlusion_ms: int = 100):
        self.model = no_inplace(model).to(device)
        self.device = device
        self.background = background.to(device)
        self.ig = IntegratedGradients(self.model)
        self.dl = DeepLift(self.model)
        self.gs = GradientShap(self.model)
        self.ixg = InputXGradient(self.model)
        self.gc = LayerGradCam(self.model, self.model.last_conv)
        self.occ = Occlusion(self.model)
        self.win = int(occlusion_ms * config.TARGET_FS / 1000)

    def attribute(self, method: str, x: torch.Tensor) -> np.ndarray:
        """x: [batch, 12, samples]. Returns [batch, 12, samples] for lead x
        time methods and [batch, samples] for time-only ones."""
        x = x.to(self.device).clone().requires_grad_(True)
        zeros = torch.zeros_like(x)
        if method == "IG":
            a = self.ig.attribute(x, baselines=zeros, n_steps=config.XAI_IG_STEPS, internal_batch_size=64)
        elif method == "DeepLIFT":
            a = self.dl.attribute(x, baselines=zeros)
        elif method == "GradientSHAP":
            a = self.gs.attribute(x, baselines=self.background, n_samples=16, stdevs=0.0)
        elif method == "InputXGradient":
            a = self.ixg.attribute(x)
        elif method == "Grad-CAM":
            cam = self.gc.attribute(x, relu_attributions=True)
            cam = LayerAttribution.interpolate(cam, (x.shape[-1],), interpolate_mode="linear")
            return cam.detach().float().cpu().numpy()[:, 0, :]
        elif method == "Occlusion":
            with torch.no_grad():
                a = self.occ.attribute(x.detach(), sliding_window_shapes=(x.shape[1], self.win),
                                       strides=(x.shape[1], self.win), baselines=0.0,
                                       perturbations_per_eval=64)
            return a.detach().float().cpu().numpy()[:, 0, :]     # same value in every lead
        else:
            raise ValueError(method)
        return a.detach().float().cpu().numpy()
