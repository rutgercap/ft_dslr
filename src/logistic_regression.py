from typing import Sequence
from numpy import ndarray
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prediction_loss(y_pred: ndarray, y: ndarray):
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)


class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: ndarray, y: ndarray, silent=False):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            loss = np.mean(prediction_loss(predictions, y))
            if not silent:
                print("loss:", loss)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X: ndarray) -> Sequence[int]:  
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
    
    def probabilities(self, X: ndarray) -> Sequence[float]:
        linear_pred = np.dot(X, self.weights) + self.bias
        return linear_pred

