from __future__ import annotations
import math
from typing import Tuple, Optional, Dict, Callable
import numpy as np

Shape = Tuple[int, ...]
Tensor = np.ndarray

def _default_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()

def calculate_fans(shape: Shape) -> Tuple[int, int]:
    """
    Returns (fan_in, fan_out) for a 2D weight matrix.
    For Linear layers, shape = (in_features, out_features).
    """
    if len(shape) < 2:
        raise ValueError(f"calculate_fans expects dim>=2, got {shape}")
    fan_in, fan_out = shape[0], shape[1]
    return fan_in, fan_out

# Fillers
def zeros(shape: Shape, rng: Optional[np.random.Generator] = None, dtype=np.float32) -> Tensor:
    return np.zeros(shape, dtype=dtype)

def ones(shape: Shape, rng: Optional[np.random.Generator] = None, dtype=np.float32) -> Tensor:
    return np.ones(shape, dtype=dtype)

def constant(shape: Shape, val: float, rng: Optional[np.random.Generator] = None, dtype=np.float32) -> Tensor:
    return np.full(shape, val, dtype=dtype)

def small_normal(shape: Shape, std: float = 0.01, rng: Optional[np.random.Generator] = None,
                 dtype=np.float32) -> Tensor:
    rng = _default_rng(rng)
    return rng.normal(loc=0.0, scale=std, size=shape).astype(dtype)

# Advanced initializers, with variance scaling
def glorot_uniform(shape: Shape, rng: Optional[np.random.Generator] = None,
                   dtype=np.float32) -> Tensor:
    """
    Xavier/Glorot uniform: U(-a, a), a = sqrt(6 / (fan_in + fan_out))
    Best for tanh/sigmoid (zero-mean, keeps variance).
    """
    rng = _default_rng(rng)
    fan_in, fan_out = calculate_fans(shape)
    a = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=shape).astype(dtype)

def glorot_normal(shape: Shape, rng: Optional[np.random.Generator] = None,
                  dtype=np.float32) -> Tensor:
    """
    Xavier/Glorot normal: N(0, sqrt(2/(fan_in+fan_out))).
    """
    rng = _default_rng(rng)
    fan_in, fan_out = calculate_fans(shape)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=shape).astype(dtype)

def he_uniform(shape: Shape, rng: Optional[np.random.Generator] = None,
               dtype=np.float32) -> Tensor:
    """
    He/Kaiming uniformly distributed for ReLU/leaky ReLU: U(-a, a), a = sqrt(6/fan_in).
    """
    rng = _default_rng(rng)
    fan_in, _ = calculate_fans(shape)
    a = math.sqrt(6.0 / fan_in)
    return rng.uniform(-a, a, size=shape).astype(dtype)

def he_normal(shape: Shape, rng: Optional[np.random.Generator] = None,
              dtype=np.float32) -> Tensor:
    """
    He/Kaiming normally distributed for ReLU/leaky ReLU: N(0, sqrt(2/fan_in)).
    """
    rng = _default_rng(rng)
    fan_in, _ = calculate_fans(shape)
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(dtype)

def lecun_uniform(shape: Shape, rng: Optional[np.random.Generator] = None,
                  dtype=np.float32) -> Tensor:
    """
    LeCun uniform for SELU: U(-a, a), a = sqrt(3/fan_in).
    """
    rng = _default_rng(rng)
    fan_in, _ = calculate_fans(shape)
    a = math.sqrt(3.0 / fan_in)
    return rng.uniform(-a, a, size=shape).astype(dtype)

def lecun_normal(shape: Shape, rng: Optional[np.random.Generator] = None,
                 dtype=np.float32) -> Tensor:
    """
    LeCun normal for SELU: N(0, sqrt(1/fan_in)).
    """
    rng = _default_rng(rng)
    fan_in, _ = calculate_fans(shape)
    std = math.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(dtype)


def orthogonal(shape: Shape, gain: float = 1.0, rng: Optional[np.random.Generator] = None,
               dtype=np.float32) -> Tensor:
    """
    Orthogonal init (for square-ish matrices). Preserves norm; good for deep nets.
    If rows < cols, we orthogonalize on the larger dimension then slice.
    """
    rng = _default_rng(rng)
    if len(shape) != 2:
        raise ValueError("orthogonal expects a 2D shape")
    rows, cols = shape
    # Generate a random Gaussian matrix
    A = rng.normal(0.0, 1.0, size=(rows, cols)).astype(dtype)
    # SVD
    u, _, vt = np.linalg.svd(A, full_matrices=False)
    Q = u if rows >= cols else vt.T
    Q = Q.astype(dtype)
    return (gain * Q)[:rows, :cols]



def gain_for_activation(name: str, param: Optional[float] = None) -> float:
    """
    Approximate 'gain' used commonly with orthogonal or variance corrections.
    """
    key = name.lower()
    if key in ("linear", "identity", "sigmoid"):
        return 1.0
    if key in ("tanh",):
        return 5.0 / 3.0  # PyTorch convention
    if key in ("relu",):
        return math.sqrt(2.0)
    if key in ("leaky_relu", "lrelu"):
        negative_slope = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1.0 + negative_slope**2))
    if key in ("selu",):
        return 1.0  # Selu expects LeCun init
    return 1.0


get_initializer: Dict[str, Callable[..., Tensor]] = {
    "small_normal": small_normal,
    "glorot_uniform": glorot_uniform,
    "glorot_normal": glorot_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "lecun_uniform": lecun_uniform,
    "lecun_normal": lecun_normal,
    "orthogonal": orthogonal,
}   