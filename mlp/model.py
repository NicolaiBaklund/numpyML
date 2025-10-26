from .types import Tensor, Layer
from typing import List, Tuple, Optional
import numpy as np
from .activations import Sigmoid, ReLU
from .layers import Dense


class Sequential:
    def __init__(self, layers: List[Layer]) -> None:

        self.layers: List[Layer] = layers
        self.is_training: bool = True

        # Validate network shape compatibility
        self._validate_network(layers, input_dim=layers[0].in_features)


    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training)
        return out
    
    def backward(self, dY: Tensor) -> Tensor:
        grad = dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.params())
        return params
    
    def grads(self) -> List[Tensor]:
        grads: List[Tensor] = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads
    
    def train(self) -> None:
        self.is_training = True
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        self.is_training = False
        for layer in self.layers:
            layer.eval()
    
    def _validate_network(self, layers: List[Layer], input_dim: int):
        X = np.zeros((1, input_dim), dtype=np.float32)  
        for i, layer in enumerate(layers):
            try:
                X = layer.forward(X, training=False)
            except Exception:
                raise ValueError(
                    f"Shape mismatch at layer {i} ({layer.__class__.__name__}). "
                    f"Expected something compatible with shape {X.shape}."
                )
