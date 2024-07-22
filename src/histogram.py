import sys
import pandas as pd

def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df

def draw_histogram(dataframe: pd.DataFrame):
    dataframe.hist(bins=4)
    print(dataframe.index)

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python describe.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if not path_to_dataset.endswith(".csv"):
        print("Invalid file format. Only CSV files are supported.")
        exit(1)
    dataframe = read_file(path_to_dataset)
    draw_histogram(dataframe)

if __name__ == '__main__':
    main()
    