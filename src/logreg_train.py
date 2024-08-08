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


def get_X_and_Y(path_to_dataset: str) -> tuple[np.ndarray, np.ndarray]:
    df = read_file(path_to_dataset)
    df.dropna(inplace=True)
    df.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
        inplace=True,
    )

    X = df.drop(columns=["Hogwarts House"])

    X = (X - X.mean()) / X.std()
    Y = df["Hogwarts House"]

    x, y = X.to_numpy(), Y.to_numpy()
    return x, y


def binary_vector(y: np.ndarray, correct_house: str) -> np.ndarray:
    result = [1 if house == correct_house else 0 for house in y]
    return np.array(result)


def house_name_to_index(y: np.ndarray):
    houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    mapping = {value: idx for idx, value in enumerate(houses)}
    index_array = np.array([mapping[house] for house in y])
    return index_array


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python logreg_train.py <path/to/train_dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only CSV files are supported.")
        exit(1)
    X_train, y = get_X_and_Y(path_to_dataset)

    multi = MultiClassRegression(4)
    multi.train(X_train, house_name_to_index(y), silent=True)
    multi.save_weights("weights.json")


if __name__ == "__main__":
    main()
