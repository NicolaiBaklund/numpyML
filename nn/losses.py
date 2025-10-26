from .types import Tensor
from typing import Optional, List
import numpy as np



class CrossEntropyLossWithLogits:
    """ 
    Cross-Entropy Loss with Softmax
    Combines softmax activation and cross-entropy loss in a numerically stable way.
    L = -sum(y_true * log(softmax(logits)))

    Used in classifiers.
    """
    def __init__(self) -> None:
        self._probs: Optional[Tensor] = None
        self._y_true: Optional[Tensor] = None
        self._batch: Optional[int] = None
    
    def forward(self, logits: Tensor, y: Tensor, training: bool = True) -> float:
        """
        logits : (B, C) raw scores from model
        y      : (B,) class indices OR (B, C) one-hot
        returns scalar loss
        """

        # Stability shift so exp doesn't explode
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        exp_vals = np.exp(shifted)
        sums = np.sum(exp_vals, axis=1, keepdims=True)

        # softmax probabilities
        probs = exp_vals / sums   # (B, C)

        # log-softmax
        log_probs = shifted - np.log(sums) # (B, C)

        B = logits.shape[0]

        # compute loss
        if y.ndim == 1:  # class index labels
            # select log-prob of correct class, index in log_probs = y_true[B_i]
            loss = -log_probs[np.arange(B), y].mean() # mean over batch

        else:  # one-hot labels
            # select log-prob of correct class, index in log_probs = y_true[B_i].index(1)
            # instead do element-wise multiply and sum over classes (all are 0 except correct class)
            loss = -(y * log_probs).sum(axis=1).mean() # mean over batch

        # caching
        if training:
            self._probs = probs  # softmax output
            self._y_true = y
            self._batch = B

        return float(loss)

    def backward(self) -> Tensor:
        """
        returns gradient of loss w.r.t. logits, shape (B, C)
        dL/dlogits = (softmax(logits) - y_true) / B
        """
        probs = self._probs
        y_true = self._y_true
        B = self._batch

        assert probs is not None and y_true is not None and B is not None, "CrossEntropyLoss.backward called before forward."

        dlogits = probs.copy()  # (B, C)
        dlogits -= y_true
        dlogits /= B
        return dlogits
    

class MSELoss:
    """
    Mean Squared Error Loss (per-batch average)
    L = mean((y_pred - y_true)^2) / 2
    dL/dy_pred = (y_pred - y_true) / batch_size
    """
    def __init__(self) -> None:
        self._y_true = None
        self._y_pred = None
        self._batch = None

    def forward(self, y_pred: Tensor, y_true: Tensor, training: bool = True) -> float:
        loss = np.mean((y_pred - y_true) ** 2) / 2
        if training:
            self._y_true = y_true
            self._y_pred = y_pred
            self._batch = y_pred.shape[0]
        return float(loss)

    def backward(self) -> Tensor:
        assert self._y_true is not None and self._y_pred is not None, "MSELoss.backward called before forward."
        return (self._y_pred - self._y_true) / self._batch


class BinaryCrossEntropyWithLogits:
    """Binary cross-entropy with logits (numerically stable).

    Expected shapes:
      - logits: (B,) or (B, 1)
      - y:      (B,) or (B, 1) with values in {0, 1} or [0, 1]

    forward returns a scalar float loss. backward returns dL/dlogits with the
    same shape as `logits` was provided (column vector if (B,1) was passed).
    """

    def __init__(self) -> None:
        self._logits: Optional[Tensor] = None
        self._y: Optional[Tensor] = None
        self._batch: Optional[int] = None
        self._return_column: bool = True

    def forward(self, logits: Tensor, y: Tensor, training: bool = True) -> float:
        # Remember whether logits were column-shaped to mirror it in backward
        self._return_column = (logits.ndim == 2 and logits.shape[1] == 1)

        # Flatten to (B,) for stable math
        z = logits.reshape(-1)
        y = y.reshape(-1).astype(z.dtype)

        # Stable BCE with logits: max(z,0) - z*y + log(1 + exp(-|z|))
        max_z0 = np.maximum(z, 0.0)
        log_term = np.log1p(np.exp(-np.abs(z)))
        per_example = max_z0 - z * y + log_term
        loss = float(per_example.mean())

        if training:
            self._logits = z
            self._y = y
            self._batch = z.shape[0]

        return loss

    def backward(self) -> Tensor:
        z = self._logits
        y = self._y
        B = self._batch
        assert z is not None and y is not None and B is not None, (
            "BinaryCrossEntropyWithLogits.backward called before forward."
        )

        # dL/dz = (sigmoid(z) - y) / B
        sig = 1.0 / (1.0 + np.exp(-z))
        grad = (sig - y) / B

        if self._return_column:
            return grad.reshape((-1, 1))
        return grad

 