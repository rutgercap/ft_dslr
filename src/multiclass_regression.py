import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy import array, ndarray

from logistic_regression import LogisticRegression


def binary_vector_index(y: ndarray, index: int) -> ndarray:
    result = [1 if i == index else 0 for i in y]
    return array(result)


class MultiClassRegression:
    regressions: Sequence[LogisticRegression]

    def __init__(self, output_classes: int):
        self.regressions = [LogisticRegression() for _ in range(output_classes)]

    def load_weights(self, weights: ndarray, biases: ndarray) -> "MultiClassRegression":
        for i in range(len(self.regressions)):
            self.regressions[i].load_weights(weights[i], biases[i])
        return self

    def train(
        self,
        X: ndarray,
        y: ndarray,
        epochs=1000,
        learning_rate=0.001,
        silent=False,
        batch_size: Optional[int] = None,
    ) -> float:
        losses = []
        now = datetime.now()
        if not silent:
            print("Starting training...")
        for i, regression in enumerate(self.regressions):
            y_each_house = binary_vector_index(y, i)
            loss = regression.fit(
                X,
                y_each_house,
                silent=True,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            losses.append(loss)
        average_loss = np.mean(losses)
        if not silent:
            end = datetime.now()
            training_time = end - now
            print(
                f"Training completed in {training_time.total_seconds():.4f} seconds. Loss: {average_loss}"
            )
        return average_loss

    def softmax_to_class(self, softmax: ndarray, classes: Sequence[str]) -> ndarray:
        result_in_classes = []
        for r in softmax.T:
            max_index = np.argmax(r)
            result_in_classes.append(classes[max_index])
        return result_in_classes

    def predict(self, X: ndarray, classes: Sequence[str]) -> ndarray:
        predictions = [regression.probabilities(X) for regression in self.regressions]
        softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=0)
        return self.softmax_to_class(softmax, classes)

    def save_weights(self, path: Path):
        weights = [
            {"weights": x.get_weights(), "bias": x.get_bias()} for x in self.regressions
        ]
        with open(path, "w+") as file:
            json.dump(weights, file)
