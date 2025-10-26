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


from .types import Tensor
from typing import Optional
import numpy as np

class BinaryCrossEntropyWithLogits:
    """Binary cross-entropy with logits (numerically stable).

    Forward expects:
      logits: shape (B,) or (B, 1)  (raw model outputs)
      y:      shape (B,) or (B, 1)  (0 or 1 floats)

    Returns scalar loss (float). Backward returns gradient shape same as logits.
    """
    def __init__(self) -> None:
        self._logits: Optional[Tensor] = None
        self._y: Optional[Tensor] = None
        self._batch: Optional[int] = None

    def forward(self, logits: Tensor, y: Tensor, training: bool = True) -> float:
        # flatten shapes to (B,)
        logits = logits.reshape(-1)
        y = y.reshape(-1).astype(logits.dtype)

        # numerically stable per-element BCE with logits
        # loss_i = max(z,0) - z*y + log(1 + exp(-abs(z)))
        z = logits
        max_z0 = np.maximum(z, 0.0)
        log_term = np.log1p(np.exp(-np.abs(z)))  # log(1+exp(-|z|))
        per_example = max_z0 - z * y + log_term
        loss = per_example.mean()

        if training:
            self._logits = z
            self._y = y
            self._batch = z.shape[0]

        return float(loss)

    def backward(self) -> Tensor:
        assert self._logits is not None and self._y is not None and self._batch is not None, \
            "BinaryCrossEntropyWithLogits.backward called before forward."
        z = self._logits
        y = self._y
        B = self._batch

        # sigmoid(z) - y  (shape (B,))
        sig = 1.0 / (1.0 + np.exp(-z))
        grad = (sig - y) / B
        return grad.reshape((-1, 1))  # reshape to column if upstream expects (B,1)