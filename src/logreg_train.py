import sys
from logistic_regression import LogisticRegression
import numpy as np
import pandas as pd


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def get_df(path_to_dataset: str):
    df = read_file(path_to_dataset)
    df.dropna(inplace=True)
    df.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
        inplace=True,
    )
    return df


def get_X_and_Y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=["Hogwarts House"])

    X = (X - X.mean()) / X.std()
    Y = df["Hogwarts House"]

    # delete it later!
    X.drop(X.index[5:], inplace=True)
    Y.drop(Y.index[5:], inplace=True)

    x, y = X.to_numpy(), Y.to_numpy()
    return x, y


def binary_vector(y: np.ndarray, correct_house: str) -> np.ndarray:
    result = [1 if house == correct_house else 0 for house in y]
    return np.array(result)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python logreg_train.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only CSV files are supported.")
        exit(1)
    df_train = get_df(path_to_dataset)
    X_train, y = get_X_and_Y(df_train)

    y_Ravenclaw = binary_vector(y, "Ravenclaw")
    clf = LogisticRegression(learning_rate=0.01)
    clf.fit(X_train, y_Ravenclaw)
    return 0

if __name__ == "__main__":
    main()
