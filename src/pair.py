import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def ft_pair(
    subjects: list[str],
    df: pd.DataFrame,
):
    houses = ["Hufflepuff", "Gryffindor", "Ravenclaw", "Slytherin"]
    colors = ["orange", "red", "blue", "green"]
    color_dict = {house: color for house, color in zip(houses, colors)}
    color_list = [color_dict[house] for house in df["Hogwarts House"]]

    figure, axis = plt.subplots(13, 13)
    plt.subplots_adjust(hspace=0.5, left=0, right=1, top=0.9, bottom=0)
    figure.suptitle("Pair plot")
    for x, x_subject in enumerate(subjects):
        for y, y_subject in enumerate(subjects):
            if y_subject == x_subject:
                continue
            df.plot.scatter(
                x=x_subject,
                y=y_subject,
                s=5,
                color=color_list,
                ax=axis[x, y],
                alpha=0.5,
            )

    legend_handles = [
        mpatches.Patch(color=color_dict["Hufflepuff"], label="Hufflepuff"),
        mpatches.Patch(color=color_dict["Gryffindor"], label="Gryffindor"),
        mpatches.Patch(color=color_dict["Ravenclaw"], label="Ravenclaw"),
        mpatches.Patch(color=color_dict["Slytherin"], label="Slytherin"),
    ]
    figure.legend(handles=legend_handles, loc="upper left")
    plt.show()


def draw_pair(df: pd.DataFrame):
    subjects = df.columns.to_list()
    for items in [
        "Index",
        "First Name",
        "Last Name",
        "Birthday",
        "Best Hand",
    ]:
        subjects.remove(items)

    df = df.filter(items=subjects)
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = (numeric_df - numeric_df.mean()) / numeric_df.std()

    sns.pairplot(df, hue="Hogwarts House", height=3)
    plt.show()

    # ft_pair(subjects, df)


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
    draw_pair(df)


if __name__ == "__main__":
    main()
