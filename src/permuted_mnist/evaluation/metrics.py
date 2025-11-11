from __future__ import annotations
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float((y_true == y_pred).mean())