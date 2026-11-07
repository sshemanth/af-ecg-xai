"""Networks for 12-lead ECGs.

ResNet1d: multichannel 1D CNN with residual blocks (the main model).
InceptionTime1d: parallel convolutions of three widths at full time
resolution, a structurally different network used to check that the
findings do not depend on one architecture.

Both end in global average pooling and one output (a logit; sigmoid gives
the AF probability), so they accept any length (train on 9 s crops, test
on 10 s). Each keeps its last convolutional stage as `self.last_conv` so
Grad-CAM can hook into it.
"""
import torch
from torch import nn

from src import config


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ResNet1d(nn.Module):
    def __init__(self, n_leads: int = len(config.LEADS),
                 widths: list[int] = config.RESNET_WIDTHS,
                 blocks: int = config.RESNET_BLOCKS,
                 kernel: int = config.RESNET_KERNEL,
                 dropout: float = config.DROPOUT):
        super().__init__()
        # stem: 500 Hz -> 250 Hz
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages, c = [], widths[0]
        for i, w in enumerate(widths):
            layers = []
            for b in range(blocks):
                stride = 2 if (b == 0 and i > 0) else 1   # each new stage halves time
                layers.append(ResBlock(c, w, kernel, stride, dropout))
                c = w
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, 1))

    @property
    def last_conv(self) -> nn.Module:
        """Last convolutional stage, for Grad-CAM. A property rather than a
        second reference, so the module is registered only once."""
        return self.stages[-1]

    def forward(self, x):                                # x: [batch, 12, samples]
        return self.head(self.stages(self.stem(x))).squeeze(-1)   # logits [batch]


class InceptionModule(nn.Module):
    """Bottleneck, then three convolutions of different widths and a
    max-pool branch, side by side; outputs are stacked."""

    def __init__(self, c_in: int, filters: int, kernels: list[int], bottleneck: int):
        super().__init__()
        self.bottleneck = (nn.Conv1d(c_in, bottleneck, 1, bias=False)
                           if c_in > 1 else nn.Identity())
        c_b = bottleneck if c_in > 1 else c_in
        self.convs = nn.ModuleList([nn.Conv1d(c_b, filters, k, padding=k // 2, bias=False)
                                    for k in kernels])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(c_in, filters, 1, bias=False))
        self.bn = nn.BatchNorm1d(filters * (len(kernels) + 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([conv(b) for conv in self.convs] + [self.pool(x)], dim=1)
        return self.act(self.bn(out))


class InceptionStage(nn.Module):
    """Three inception modules with a residual link around them."""

    def __init__(self, c_in: int, filters: int, kernels: list[int], bottleneck: int):
        super().__init__()
        c_out = filters * (len(kernels) + 1)
        mods, c = [], c_in
        for _ in range(3):
            mods.append(InceptionModule(c, filters, kernels, bottleneck))
            c = c_out
        self.mods = nn.Sequential(*mods)
        self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.mods(x) + self.skip(x))


class InceptionTime1d(nn.Module):
    def __init__(self, n_leads: int = len(config.LEADS),
                 filters: int = config.INCEPTION_FILTERS,
                 depth: int = config.INCEPTION_DEPTH,
                 kernels: list[int] = config.INCEPTION_KERNELS,
                 bottleneck: int = config.INCEPTION_BOTTLENECK,
                 dropout: float = config.DROPOUT):
        super().__init__()
        if depth % 3:
            raise ValueError("INCEPTION_DEPTH must be a multiple of 3")
        c_out = filters * (len(kernels) + 1)
        # stem: 500 Hz -> 125 Hz (strided conv, then max-pool); the kernels
        # then span about 312, 152 and 72 ms, enough for P-waves and f-waves
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, c_out, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.stages = nn.Sequential(*[InceptionStage(c_out, filters, kernels, bottleneck)
                                      for _ in range(depth // 3)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_out, 1))

    @property
    def last_conv(self) -> nn.Module:
        return self.stages[-1]

    def forward(self, x):
        return self.head(self.stages(self.stem(x))).squeeze(-1)


def build_model(arch: str = "resnet", dropout: float = config.DROPOUT) -> nn.Module:
    if arch == "resnet":
        return ResNet1d(dropout=dropout)
    if arch == "inception":
        return InceptionTime1d(dropout=dropout)
    raise ValueError(f"Unknown architecture: {arch} (choose from {config.ARCHS})")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
