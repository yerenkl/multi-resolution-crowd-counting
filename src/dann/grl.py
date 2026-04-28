import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


class GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


def ganin_alpha_schedule(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / total_epochs
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
