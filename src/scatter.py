import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import numpy as np
from itertools import combinations

def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df

def draw_scatter(df: pd.DataFrame):
    subjects = df.columns.to_list()
    for items in ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday','Best Hand']:
        subjects.remove(items)
    colors = [
    "#ff355e", "#fd5b78", "#ff6037",
    "#ff9966", "#ff9933", "#ffcc33",
    "#ffff66", "#ccff00", "#66ff66",
    "#aaf0d1", "#16d0cb", "#50bfe6",
    "#9c27b0"
    ]

    color_dict = {
        subject: color
        for subject,color in zip(subjects, colors)
    }

    
    df = df.select_dtypes(include=[np.number])
    df=(df-df.mean())/df.std()

    figure, axis = plt.subplots(4, 4)
    

    # for x_subject in subjects:
        # axis[floor(i/4), i%4].set_title(f'Compare with {x_subject}')
    x_subject = 'Flying'
    for i, y_subject in enumerate(subjects):
        print(x_subject, y_subject)
        # print(color_dict[y_subject])
        if y_subject == x_subject:
            continue
        if i == 0:
            ax = df.plot.scatter(x=x_subject, y=y_subject, color = color_dict[y_subject])
        df.plot.scatter(x=x_subject, y=y_subject, color = color_dict[y_subject], ax=ax)
    
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
    df = read_file(path_to_dataset)
    draw_scatter(df)

if __name__ == '__main__':
    main()
    