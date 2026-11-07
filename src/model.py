"""1D ResNet for 12-lead ECGs.

Multichannel 1D CNN with residual blocks, global average pooling and one
output (a logit; sigmoid gives the AF probability). Global pooling means
it accepts any length, so it can train on 9 s crops and be tested on 10 s.

The last convolutional stage is kept as `self.last_conv` so Grad-CAM
can hook into it.
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
        self.last_conv = self.stages[-1]                 # for Grad-CAM
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x):                                # x: [batch, 12, samples]
        return self.head(self.stages(self.stem(x))).squeeze(-1)   # logits [batch]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
