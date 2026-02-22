# Derived from CLMR: https://github.com/Spijkervet/CLMR (Apache-2.0).
# Modifications for HOOK are licensed under Apache-2.0.
import torch
import torch.nn as nn
from .model import Model


class SampleCNN(Model):
    def __init__(self, strides):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

    def forward(self, x):
        # Accept [B, T] or [B, C, T] inputs and normalize to mono [B, 1, T].
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        out = self.sequential(x)
        return out
