import numpy as np
from typing import List, Optional
from .types import Layer, Tensor, Shape


class SGD:
    """
    Stochastic gradient descent optimizer.
    Attributes:
        parameters: List of model parameters to optimize.
        learning_rate: Learning rate for the optimizer.


    """
    def __init__(self, parameters: List[Tensor], grads: List[Tensor], learning_rate: float = 0.01):
        self.parameters: List[Tensor] = parameters
        self.grads: List[Tensor] = grads
        self.learning_rate: float = learning_rate

    def step(self) -> None:
        for p, g in zip(self.parameters, self.grads):
            p -= self.learning_rate * g
    def zero_grad(self) -> None:
        for g in self.grads:
            g.fill(0)
    
