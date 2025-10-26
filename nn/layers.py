from .types import Layer, Tensor
from typing import List, Optional, Tuple, Callable
import numpy as np
from .initializers import get_initializer

class Dense:
    def __init__(self, in_features: int, out_features: int, 
                 w_init: str = "he_normal", b_init: str = "zeros", 
                 rng: Optional[np.random.Generator] = None) -> None:
        self.in_features: Optional[int] = in_features
        self.out_features: Optional[int] = out_features
        self.is_training: bool = True

        # Initialize weights and biases
        if w_init not in get_initializer or b_init not in get_initializer:
            raise ValueError(f"Unknown initializer: {w_init} or {b_init}")
        
        w_initializer: Callable[..., Tensor] = get_initializer[w_init]
        b_initializer: Callable[..., Tensor] = get_initializer[b_init]

        self.W: Tensor = w_initializer((in_features, out_features), rng=rng)
        self.b: Tensor = b_initializer((out_features,), rng=rng)

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
        # Optional feature counts to satisfy Layer protocol
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

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
        # Optional feature counts to satisfy Layer protocol
        self.in_features: Optional[int] = None
        self.out_features: Optional[int] = None

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

