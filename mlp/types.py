import numpy as np
from typing import List, Tuple, Protocol

Tensor = np.ndarray
Shape = Tuple[int, ...]

class Layer(Protocol):
    in_features: int
    out_features: int
    is_training: bool
    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        ...
    def backward(self, dY: Tensor) -> Tensor:
        ...
    def params(self) -> List[Tensor]: 
        ...
    def grads(self) -> List[Tensor]: 
        ...
    def train(self) -> None: 
        ...
    def eval(self) -> None: 
        ...