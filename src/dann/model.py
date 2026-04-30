from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from src.dann.grl import GradientReversalLayer
from src.dann.classifier import DomainClassifier


class DANNModel(nn.Module):
    """Wraps a CLIP_EBC model with a domain-adversarial branch.

    Uses a forward hook on the crowd model's image_decoder to capture
    intermediate features without modifying the CLIP-EBC code.
    """

    def __init__(
        self,
        crowd_model: nn.Module,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.crowd_model = crowd_model
        self.grl = GradientReversalLayer()
        self.domain_classifier = DomainClassifier(
            in_channels=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self._features: Optional[Tensor] = None
        self._hook_handle = self.crowd_model.image_decoder.register_forward_hook(
            lambda module, input, output: setattr(self, "_features", output)
        )

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        self._features = None
        crowd_output = self.crowd_model(x)

        if self.training:
            assert self._features is not None
            reversed_features = self.grl(self._features)
            domain_logits = self.domain_classifier(reversed_features)
            logits, density = crowd_output
            return logits, density, domain_logits

        return crowd_output

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
