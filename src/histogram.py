import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import floor

def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df

def draw_histogram(dataframe: pd.DataFrame):
    columns = dataframe.columns.to_list()
    for items in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday','Best Hand']:
        columns.remove(items)

    figure, axis = plt.subplots(4, 4)
    plt.subplots_adjust(hspace=0.5)

    for i, subject in enumerate(columns):
        H = dataframe.loc[dataframe['Hogwarts House'] == 'Hufflepuff', subject]
        G = dataframe.loc[dataframe['Hogwarts House'] == 'Gryffindor', subject]
        R = dataframe.loc[dataframe['Hogwarts House'] == 'Ravenclaw', subject]
        S = dataframe.loc[dataframe['Hogwarts House'] == 'Slytherin', subject]

        axis[floor(i/4), i%4].hist(H, alpha=0.5, label= 'Hufflepuff', bins=100)
        axis[floor(i/4), i%4].hist(G, alpha=0.5, label= 'Gryffindor', bins=100)
        axis[floor(i/4), i%4].hist(R, alpha=0.5, label= 'Ravenclaw', bins=100)
        axis[floor(i/4), i%4].hist(S, alpha=0.5, label= 'Slytherin', bins=100)

        axis[floor(i/4), i%4].set_title(f'{subject} Distribution by House')
        axis[floor(i/4), i%4].set_xlabel(subject)
        axis[floor(i/4), i%4].set_ylabel('Frequency')
    
    figure.legend(['Hufflepuff', 'Gryffindor', 'Ravenclaw', 'Slytherin'], loc="upper right") 
    plt.show()


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
    