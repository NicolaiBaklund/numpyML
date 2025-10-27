from .types import Tensor, Layer
from typing import List, Tuple, Optional
import numpy as np
from .activations import Sigmoid, ReLU
from .layers import Dense, Dropout, Flatten
from typing import Dict, Any
import pickle


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
        return grad # derivative of loss w.r.t input X, rarely used
    
    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def gradients(self) -> List[Tensor]:
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
    
    def _validate_network(self, layers: List[Layer], input_dim: Optional[int]):
        if input_dim is None:
            raise ValueError("Input dimension is None, cannot validate network.")
        X = np.zeros((1, input_dim), dtype=np.float32)  
        for i, layer in enumerate(layers):
            try:
                X = layer.forward(X, training=False)
            except Exception:
                raise ValueError(
                    f"Shape mismatch at layer {i} ({layer.__class__.__name__}). "
                    f"Expected something compatible with shape {X.shape}."
                )

    # --- serialization / copy helpers ---
    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of architecture + parameters."""
        layers_state = []
        for layer in self.layers:
            entry: Dict[str, Any] = {
                "class": layer.__class__.__name__,
                "config": getattr(layer, "get_config", lambda: {})(),
                "state": getattr(layer, "state_dict", lambda: {})(),
            }
            layers_state.append(entry)
        return {"layers": layers_state}

    def save(self, path: str) -> None:
        """Save model architecture + parameters to a file (pickle)."""
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Sequential":
        """Reconstruct a Sequential from a state dict (as returned by state_dict())."""
        registry = {
            "Dense": Dense,
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "Dropout": Dropout,
            "Flatten": Flatten,
        }
        layers: List[Layer] = []
        for layer_entry in state["layers"]:
            name = layer_entry["class"]
            config = layer_entry.get("config", {}) or {}
            state_blob = layer_entry.get("state", {}) or {}
            if name not in registry:
                raise ValueError(f"Unknown layer class in state: {name}")
            LayerCls = registry[name]
            kwargs = {k: v for k, v in config.items() if v is not None}
            layer = LayerCls(**kwargs) if kwargs else LayerCls()
            load_fn = getattr(layer, "load_state_dict", None)
            if callable(load_fn):
                load_fn(state_blob)
            layers.append(layer)
        return cls(layers)

    @classmethod
    def load(cls, path: str) -> "Sequential":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls.from_state_dict(state)

    def copy(self) -> "Sequential":
        """Return a deep copy of the network (new objects, same parameters)."""
        state = self.state_dict()
        return self.from_state_dict(state)
