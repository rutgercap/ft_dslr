import sys
import pandas as pd
import numpy as np

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
    X = (X - X.mean()) / X.std()
    return X.to_numpy()

def get_weights(path: str) -> np.ndarray:
    weights = np.fromfile(path)
    print(weights)

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python logreg_predict.py <weights.numpy> <path/to/dataset_test.csv> ")
        exit(1)
    path_to_weight = args[0]
    path_to_dataset = args[1]
    if not path_to_weight.endswith(".json") or not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only json files are supported.")
        exit(1)
    X_train = get_X_test(path_to_dataset)
    weights, biases = get_weights(path_to_weight)
    return 0
    multi = MultiClassRegression(4).load_weights(weights, biases)
    houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    
    result = multi.predict(np.array(X_train), houses, weights)
    print(result)
    # make a file 

    # houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    # result = multi.predict(np.array(X_train), houses)
    # print(result)
# $> cat houses.csv
# Index,Hogwarts House
# 0,Gryffindor
# 1,Hufflepuff
# 2,Ravenclaw
# 3,Hufflepuff
# 4,Slytherin
# 5,Ravenclaw
# 6,Hufflepuff
# [...]

if __name__ == "__main__":
    main()