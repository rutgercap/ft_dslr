import sys
from math import floor

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def draw_scatter(df: pd.DataFrame):
    subjects = df.columns.to_list()
    for items in [
        "Index",
        "Hogwarts House",
        "First Name",
        "Last Name",
        "Birthday",
        "Best Hand",
    ]:
        subjects.remove(items)
    houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    colors = ["orange", "red", "blue", "green"]
    color_dict = {house: color for house, color in zip(houses, colors)}
    color_list = [color_dict[house] for house in df["Hogwarts House"]]

    df = df.select_dtypes(include=[np.number])
    df = (df - df.mean()) / df.std()

    for x_subject in subjects:
        figure, axis = plt.subplots(3, 4)
        figure.suptitle(f"Compare with {x_subject}")
        plt.subplots_adjust(hspace=0.5)
        i = 0
        for y_subject in subjects:
            if y_subject == x_subject:
                continue
            df.plot.scatter(
                x=x_subject,
                y=y_subject,
                s=5,
                color=color_list,
                ax=axis[floor(i / 4), i % 4],
            )
            i += 1
        legend_handles = [
            mpatches.Patch(color=color_dict["Hufflepuff"], label="Hufflepuff"),
            mpatches.Patch(color=color_dict["Gryffindor"], label="Gryffindor"),
            mpatches.Patch(color=color_dict["Ravenclaw"], label="Ravenclaw"),
            mpatches.Patch(color=color_dict["Slytherin"], label="Slytherin"),
        ]
        figure.legend(handles=legend_handles, loc="upper left")

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


if __name__ == "__main__":
    main()
