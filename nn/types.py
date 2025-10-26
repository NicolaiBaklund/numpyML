import numpy as np
from typing import List, Tuple, Protocol, Optional

Tensor = np.ndarray
Shape = Tuple[int, ...]


class Layer(Protocol):
    """General layer protocol.

    in_features and out_features are optional because some layers (e.g. activations,
    dropout, flatten) don't have a fixed number of features. Layers that do have
    them (e.g. `Dense`) should expose integer values.
    """
    in_features: Optional[int]
    out_features: Optional[int]
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