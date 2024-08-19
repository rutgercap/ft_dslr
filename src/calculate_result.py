import sys

import pandas as pd

def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print(
            "Usage: python calculate_result.py <path/to/dataset_train.csv> <houses.csv>"
        )
        exit(1)
    original = args[0]
    prediction = args[1]
    if not original.endswith(".csv") or not prediction.endswith(".csv"):
        print("Invalid file format. Only csv files are supported.")
        exit(1)

    df_origin = read_file(original)
    df_origin.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
        inplace=True,
    )
    df_origin.dropna(inplace=True)

    df_predict = read_file(prediction)

    matches = sum(df_origin["Hogwarts House"].iloc[i] == df_predict["Hogwarts House"].iloc[i] for i in range(len(df_origin)))
    total = len(df_origin)
    percent = matches/total * 100
    print(f"Your result is {percent:.2f}% accurate")


if __name__ == "__main__":
    main()
