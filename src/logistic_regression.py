from typing import Optional, Sequence

import numpy as np
from numpy import ndarray


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prediction_loss(y_pred: ndarray, y: ndarray):
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def load_weights(self, weights: ndarray, bias: ndarray) -> "LogisticRegression":
        self.weights = weights
        self.bias = bias
        return self

    def gradient_descent(
        self,
        X: ndarray,
        y: ndarray,
        learning_rate=0.001,
        silent=False,
    ):
        shape = X.shape
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_pred)
        loss = np.mean(prediction_loss(predictions, y))
        if not silent:
            print("loss:", loss)
        dw = (1 / shape[0]) * np.dot(X.T, (predictions - y))
        db = (1 / shape[0]) * np.sum(predictions - y)

        self.weights = self.weights - learning_rate * dw
        self.bias = self.bias - learning_rate * db

    def fit(
        self,
        X: ndarray,
        y: ndarray,
        epochs=1000,
        learning_rate=0.001,
        silent=False,
        batch_size: Optional[int] = None,
    ):
        samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            if batch_size is None:
                self.gradient_descent(X, y, learning_rate, silent)
                continue
            indices = np.random.randint(0, samples, size=batch_size)
            self.gradient_descent(X[indices], y[indices], learning_rate, silent)

    def predict(self, X: ndarray) -> Sequence[int]:
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

    def probabilities(self, X: ndarray) -> Sequence[float]:
        linear_pred = np.dot(X, self.weights) + self.bias
        return linear_pred

    def get_weights(self) -> list:
        return list(self.weights)

    def get_bias(self) -> list:
        return float(self.bias)
