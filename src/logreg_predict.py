import json
import sys

import numpy as np
import pandas as pd

from multiclass_regression import MultiClassRegression


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def get_X_test(path_to_dataset: str) -> np.ndarray:
    df = read_file(path_to_dataset)
    df.dropna(inplace=True)
    df.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
        inplace=True,
    )
    X = df.drop(columns=["Hogwarts House"])
    X = (X - X.mean()) / X.std()
    return X.to_numpy()


def get_weights(path: str) -> np.ndarray:
    f = open(path)
    data = json.load(f)
    weights = np.array([item["weights"] for item in data])
    biases = np.array([item["bias"] for item in data])
    return weights, biases


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print(
            "Usage: python logreg_predict.py <weights.numpy> <path/to/dataset_test.csv> "
        )
        exit(1)
    path_to_weight = args[0]
    path_to_dataset = args[1]
    if not path_to_weight.endswith(".json") or not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only json files are supported.")
        exit(1)
    X_test = get_X_test(path_to_dataset)
    weights, biases = get_weights(path_to_weight)

    multi = MultiClassRegression(4).load_weights(weights, biases)
    houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    result = multi.predict(np.array(X_test), houses)

    indices = np.arange(len(result))
    data = np.column_stack((indices, result))
    np.savetxt(
        "houses.csv",
        data,
        fmt="%s",
        delimiter=",",
        header="Index,Hogwarts House",
        comments="",
    )


if __name__ == "__main__":
    main()
