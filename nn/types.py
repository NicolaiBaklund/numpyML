import numpy as np
from typing import List, Tuple, Protocol, Optional
from typing import Dict, Any

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
    
    # Serialization / configuration helpers
    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable configuration describing this layer (stateless)."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable mapping of parameters (typically numpy arrays)."""
        ...

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load parameters from a state dict produced by `state_dict`.

        Implementations should copy data from `state` into internal arrays.
        """
        ...