from typing import Sequence

from logistic_regression import LogisticRegression
from numpy import array, ndarray
import numpy as np

def binary_vector_index(y: ndarray, index: int) -> ndarray:
    result = [1 if i == index else 0 for i in y]
    return array(result)

class MultiClassRegression:
    regressions: Sequence[LogisticRegression]

    def __init__(self, output_classes: int, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regressions = [LogisticRegression(learning_rate=learning_rate, epochs=epochs) for _ in range(output_classes)]

    def train(self, X: ndarray, y: ndarray, silent=False):
        for i, regression in enumerate(self.regressions):
            y_each_house = binary_vector_index(y, i)
            regression.fit(X, y_each_house, silent=silent)
        
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



