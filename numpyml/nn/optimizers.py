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
    
class Adam:
    def __init__(self, parameters: List[Tensor], grads: List[Tensor], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.parameters: List[Tensor] = parameters
        self.grads: List[Tensor] = grads
        self.learning_rate: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.m: List[Tensor] = [np.zeros_like(p) for p in parameters]
        self.v: List[Tensor] = [np.zeros_like(p) for p in parameters]
        self.t: int = 0

    def step(self) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.parameters, self.grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self) -> None:
        for g in self.grads:
            g.fill(0)