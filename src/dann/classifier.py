import torch.nn as nn
from torch import Tensor


class DomainClassifier(nn.Module):

    def __init__(self, in_channels: int = 768, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: Tensor) -> Tensor:
        x = self.gap(features).squeeze(-1).squeeze(-1)
        return self.classifier(x)
