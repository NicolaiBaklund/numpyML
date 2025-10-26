from .types import Layer, Tensor
from typing import List, Optional, Tuple
import numpy as np

class Dense:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.is_training = True

        # TODO: optimize initialization
        self.W: Tensor = np.random.randn(in_features, out_features) * (1.0 / np.sqrt(in_features))
        self.b: Tensor = np.zeros((out_features,), dtype=self.W.dtype)

        # initialize grads
        self.dW: Tensor = np.zeros_like(self.W)
        self.db: Tensor = np.zeros_like(self.b)

        # input cache
        self.X: Optional[Tensor] = None

    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        # output = (batch_size, input_size) @ (input_size, output_size) + (output_size) = (batch_size, output_size)
        Y = X @ self.W + self.b
        # Cache X if training
        if training:
            self.X = X
        return Y # output

    def backward(self, dY: Tensor) -> Tensor:
        # Require cache
        X = self.X
        assert X is not None, "Dense.backward called without cached X. Call forward(training=True) before backward."

        # Gradients, copyto avoids reallocation
        # dW: (in, batch) @ (batch, out) = (in, out)
        np.copyto(self.dW, X.T @ dY)
        # db: sum over batch -> (out,)
        np.copyto(self.db, dY.sum(axis=0))
        # dX: (batch, out) @ (out, in) -> (batch, in)
        dX = dY @ self.W.T
        return dX

    def params(self) -> List[Tensor]:
        return [self.W, self.b]

    def grads(self) -> List[Tensor]:
        return [self.dW, self.db]

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False


class Dropout:
    """
    Dropout Layer
    Randomly sets a fraction of input units to zero during training.
    """
    def __init__(self, p_drop: float) -> None:
        assert 0.0 <= p_drop < 1.0, "Dropout probability must be in [0.0, 1.0)."

        self.p_drop: float = p_drop
        self.is_training: bool = True
        self.mask: Optional[Tensor] = None  # Cache

    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        if training:
            # Create dropout mask, if value < p_drop, set to 0
            # Scale to keep expected value the same, dont want the weights to learn smaller values
            self.mask = (np.random.rand(*X.shape) >= self.p_drop).astype(X.dtype) / (1.0 - self.p_drop)
            return X * self.mask  # Apply mask
        else:
            return X  # No dropout during evaluation
        
    def backward(self, dY: Tensor) -> Tensor:
        mask = self.mask
        assert mask is not None, "Dropout.backward called without cached mask. Call forward(training=True) before backward."
        dX = dY * mask  # Apply mask, activations that were dropped need zero grad.
        return dX
    
    def params(self) -> List[Tensor]:
        return []  # No parameters
    
    def grads(self) -> List[Tensor]:
        return []  # No gradients
    
    def train(self) -> None:
        self.is_training = True
    
    def eval(self) -> None:
        self.is_training = False


class Flatten:
    """
    Flatten Layer
    Reshapes input to (batch_size, -1)
    """
    def __init__(self) -> None:
        self.is_training: bool = True
        self.input_shape: Optional[Tuple[int, ...]] = None  # Cache

    def forward(self, X: Tensor, training: bool = True) -> Tensor:
        if training:
            self.input_shape = X.shape  # Cache original shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)  # Flatten all but batch dimension

    def backward(self, dY: Tensor) -> Tensor:
        input_shape = self.input_shape
        assert input_shape is not None, "Flatten.backward called without cached input shape. Call forward(training=True) before backward."
        return dY.reshape(input_shape)  # Reshape back to original shape

    def params(self) -> List[Tensor]:
        return []  # No parameters

    def grads(self) -> List[Tensor]:
        return []  # No gradients

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False

