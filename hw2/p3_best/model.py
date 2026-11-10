"""Exact mini_gap architecture used by the winning challenger."""

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class MiniGAPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (16, 32, 64)
        layers = [nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                  nn.BatchNorm2d(widths[0]), nn.ReLU(), nn.MaxPool2d(2)]
        previous = widths[0]
        for stage, width in enumerate(widths):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(previous, width, stride))
                previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(widths[-1], 5),
        )

    def forward(self, x):
        return self.classifier(self.features((x - 0.5) / 0.25))
