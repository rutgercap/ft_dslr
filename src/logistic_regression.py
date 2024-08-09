from typing import Sequence

import numpy as np
from numpy import array, ndarray


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

    def fit(
        self, X: ndarray, y: ndarray, epochs=1000, learning_rate=0.001, silent=False
    ):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            loss = np.mean(prediction_loss(predictions, y))
            if not silent:
                print("loss:", loss)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

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
