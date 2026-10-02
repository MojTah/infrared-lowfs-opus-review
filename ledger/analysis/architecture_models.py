"""Native-pixel architecture controls; frozen benchmark source stays unchanged.

ResNet18 uses torchvision's existing implementation with no downloaded weights.
Its 3x3 stride-one stem, omitted max-pool and four-output head are adaptations,
not an exact reproduction of Taheri et al. 2024. Neither control resizes pixels.
Sources: https://arxiv.org/html/2410.12084v1 (section 4),
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from benchmark import learning

ARCHITECTURES = ("compact_cnn", "resnet18")


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1))

    def forward(self, image):
        return torch.relu(image + self.layers(image))


def build(architecture, input_size):
    if architecture not in ARCHITECTURES or input_size not in (256, 512):
        raise ValueError("Require a declared architecture and one/two native 16x16 channels")
    channels = input_size // 256
    if architecture == "compact_cnn":
        network = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), ResidualBlock(),
            nn.AvgPool2d(2), ResidualBlock(), nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 4))
    else:
        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
        network.fc = nn.Linear(512, 4)
    return nn.Sequential(nn.Unflatten(1, (channels, 16, 16)), network)


def fit(architecture, x, labels, validation_x, validation_labels,
        validation_parents, seed, device, guard, *, epochs=100, batch_size=128):
    """Match baseline Adam/MSE, repeated seeds and validation-only early stopping.

    Arrays are train-fitted asinh features. Outputs are scaled by the existing
    four-mode LABEL_SCALE, so the frozen prediction/calibration helpers apply.
    The caller owns cumulative deadline/resources; guard runs at each batch.
    """
    if (x.ndim != 2 or x.shape[1] not in (256, 512) or len(x) == 0
            or validation_x.ndim != 2 or validation_x.shape[1] != x.shape[1]
            or len(validation_x) == 0 or labels.shape != (len(x), 4)
            or validation_labels.shape != (len(validation_x), 4)
            or len(validation_parents) != len(validation_x)):
        raise ValueError("Training/validation dimensions disagree")
    if not all(np.isfinite(a).all() for a in (x, labels, validation_x, validation_labels)):
        raise ValueError("Nonfinite training/validation input")
    if not 1 <= epochs <= 100 or not 1 <= batch_size <= 128:
        raise ValueError("Epoch/batch bounds exceed the matched protocol")
    guard()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = build(architecture, x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    ty = torch.from_numpy((labels / learning.LABEL_SCALE).astype(np.float32)).to(device)
    vx = torch.from_numpy(np.asarray(validation_x, dtype=np.float32)).to(device)
    vy = torch.from_numpy((validation_labels / learning.LABEL_SCALE).astype(np.float32)).to(device)
    vw = torch.from_numpy(learning._parent_weights(validation_parents).astype(np.float32)).to(device)
    best, best_state, stale, history = math.inf, None, 0, []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), math.ceil(len(x) / batch_size)):
            guard()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(tx[ids]) - ty[ids]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            guard()
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for first in range(0, len(vx), 512):
                guard()
                residual = model(vx[first:first + 512]) - vy[first:first + 512]
                validation_loss += float(torch.sum(torch.mean(residual ** 2, dim=1)
                                                    * vw[first:first + 512]).item())
                guard()
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite validation loss")
        history.append(validation_loss)
        if validation_loss < best:
            best = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    guard()
    model.load_state_dict(best_state, strict=True)
    return model, {"architecture": architecture, "seed": seed,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "best_validation_standardized_mse": best, "epochs": len(history),
                   "validation_history": history, "elapsed_seconds": time.monotonic() - started}


def self_check():
    """CPU-only interface/gradient/fit checks, not a scientific training result."""
    torch.set_num_threads(1)
    sizes = {}
    for architecture in ARCHITECTURES:
        for input_size in (256, 512):
            model = build(architecture, input_size)
            prediction = model(torch.randn(2, input_size))
            assert prediction.shape == (2, 4) and torch.isfinite(prediction).all()
            prediction.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            sizes[f"{architecture}/{input_size}"] = sum(p.numel() for p in model.parameters())
    assert sizes["compact_cnn/256"] < 50_000 and sizes["compact_cnn/512"] < 50_000
    for architecture, input_size in (("unknown", 256), ("resnet18", 100)):
        try:
            build(architecture, input_size)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid model input accepted")
    rng = np.random.default_rng(129)
    x = rng.normal(size=(8, 256)).astype(np.float32)
    labels = rng.normal(size=(8, 4)) * learning.LABEL_SCALE
    calls = []
    model, report = fit("compact_cnn", x, labels, x[:4], labels[:4],
                        np.array(["a", "a", "b", "b"]), 11, torch.device("cpu"),
                        lambda: calls.append(1), epochs=2, batch_size=4)
    assert report["epochs"] == 2 and math.isfinite(report["best_validation_standardized_mse"])
    assert model(torch.zeros(2, 256)).shape == (2, 4) and len(calls) >= 10
    def stop():
        raise TimeoutError("self-check deadline")
    try:
        fit("compact_cnn", x, labels, x[:4], labels[:4], np.arange(4),
            11, torch.device("cpu"), stop, epochs=2)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Deadline guard was ignored")
    damaged = x.copy()
    damaged[0, 0] = np.nan
    try:
        fit("compact_cnn", damaged, labels, x[:4], labels[:4], np.arange(4),
            11, torch.device("cpu"), lambda: None, epochs=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite training input accepted")
    print({"status": "PASS_CPU_MODEL_INTERFACE_GRADIENT_AND_FIT", "parameters": sizes,
           "scope": "synthetic CPU controls only; no scientific run or GPU admission"})


if __name__ == "__main__":
    self_check()
