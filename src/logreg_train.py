import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns
import math

def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df

def formula(values: list, weights: list) -> float:
    if (len(values) +  1 != len(weights)):
        raise ValueError("values and weights have to have same length.")
    result = weights[len(values)]
    for a, b in zip(values, weights):
        result += a * b
    return result

def continuous_to_categorical(df):
    pd.cut(df.results,bins=[0,1,2,3,4],labels=['Hufflepuff','Gryffindor','Ravenclaw','Slytherin'])

# results = {
#     'H' : 0.3
#     'G' : 0.2
#     'R' : 0.1
#     'S' : 0.4 - correct
# }
# log(0.4) + log(1-0.3) + log(1-0.2) + log(1-0.1)
def calculate_loss(results: dict, correct_house: str):
    # y * log(h(x)) + (1-y) * log(1-h(x))
    sum = 0
    for key, value in results.items:
        y = key == correct_house
        sum += y * math.log(value) + (1 - y) * math.log(1 - value)
    return sum


def softmax(results):
    return (np.exp(results)/sum(np.exp(results)))

def one_hot_encoded_vector(k: str) -> list[int]:
    # return [k == "Hufflepuff",k == "Gryffindor",k == "Ravenclaw",k == "Slytherin"]
    return [0,0,0,0]

def calculate_gredient(y_preds, X, Y, m):
    X = X.reset_index(drop=True)
    X = X.T
    # X = X.reset_index(drop=True)

    gredients = []
    for subject in X.iterrows():
        i = 0
        sum = 0
        for y_pred, y in zip(y_preds, Y):
            x = subject[1][i]
            sum += y_pred[0] - (y == "Hufflepuff")
            i += 1
        gredient = sum / m * x
        gredients.append(gredient)
    return gredients


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python logreg_train.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only CSV files are supported.")
        exit(1)
    df = read_file(path_to_dataset)
    df.dropna(inplace=True)
    df.drop(columns=['Index','First Name','Last Name','Birthday', 'Best Hand'], inplace=True)
    X = df.drop(columns=['Hogwarts House'])

    X = (X-X.mean())/X.std()
    Y = df['Hogwarts House']
    m = Y.shape[0]

    X.drop(X.index[5:], inplace=True)
    Y.drop(Y.index[5:], inplace=True)

    weights_per_house = {
        house : np.array([np.random.random() for _ in range(len(X.columns) + 1)])
        for house in ['Hufflepuff','Gryffindor','Ravenclaw','Slytherin']
    }

    for _ in range(1):
        y_preds = []
        for student in X.to_numpy():
            results = {}
            for house in weights_per_house:
                results[house] = formula(student, weights_per_house[house])
            y_pred = softmax(list(results.values()))
            y_preds.append(y_pred)
        gredients = calculate_gredient(y_preds, X, Y, m)
        print(gredients)
        # weight -= - gredient * learning_speed


if __name__ == "__main__":
    main()
