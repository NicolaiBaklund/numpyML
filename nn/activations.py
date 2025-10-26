from .types import Tensor, Shape, Layer
from typing import List, Tuple, Optional
import numpy as np


class ReLU:
    """
    ReLU Activation Layer
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0
    """
    def __init__(self) -> None:
        self.is_training: bool = True
        # input cache, mask for backward, derivative of each input element
        self.mask: Tensor | None = None
        # Optional feature counts to satisfy Layer protocol
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        # ReLU activation: f(x) = max(0, x)
        Y = np.maximum(0, X)
        if training and self.is_training:
            # Cache mask for backward, 1 where X>0, else 0
            self.mask = (X > 0).astype(X.dtype)
        return Y

    def backward(self, dY: Tensor) -> Tensor:
        mask = self.mask
        assert mask is not None, "ReLU.backward called without cached mask. Call forward(training=True) before backward."
        dX = dY * mask # Hadamard product (Element wise)
        return dX

    def params(self) -> List[Tensor]:
        return [] # No parameters

    def grads(self) -> List[Tensor]:
        return [] # No gradients

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False


def sigmoid(X: Tensor) -> Tensor:
    return 1/(1+np.exp(-X))

def d_sigmoid(X: Tensor) -> Tensor:
    Y = sigmoid(X)
    return Y*(1-Y)


class Sigmoid:
    """
    Sigmoid Activation Layer
    f(x) = 1/(1+exp(-x))
    f'(x) = f(x)*(1-f(x))
    """
    def __init__(self) -> None:
        self.is_training: bool = True
        self.X: Optional[Tensor] = None # Cache
        # Optional feature counts to satisfy Layer protocol
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        # Cache input if training
        if training and self.is_training:
            self.X = X
        return sigmoid(X)
    def backward(self, dY: Tensor) -> Tensor:
        X = self.X
        assert X is not None, "Sigmoid.backward called without cached mask. Call forward(training=True) before backward."

        dA = d_sigmoid(X)
        # The derivative of the input w.r.t the output
        dX = dA * dY
        
        return dX

    def params(self) -> List[Tensor]:
        return [] # No parameters

    def grads(self) -> List[Tensor]:
        return [] # No gradients

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False

        



