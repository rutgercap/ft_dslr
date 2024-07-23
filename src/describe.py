import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from prettytable import PrettyTable


def count(values: Sequence[Any]) -> float:
    count = 0.0
    for value in values:
        if not pd.isna(value):
            count += 1
    return count


def mean(values: Sequence[Any]) -> float:
    n = 0
    sum = 0.0
    for value in values:
        if not pd.isna(value):
            n += 1
            sum += value
    if n == 0:
        return float("nan")
    return sum / n


def std(values: Sequence[Any]) -> float:
    values_mean = mean(values)
    n = 0
    sum_of_squares = 0.0
    for value in values:
        if pd.isna(value):
            continue
        n += 1
        error = value - values_mean
        error = error**2
        sum_of_squares += error
    if n == 0:
        return float("nan")
    return np.sqrt(sum_of_squares / n)


def values_min(values: Sequence[Any]) -> float:
    filtered_values = [value for value in values if not pd.isna(value)]
    if len(filtered_values) == 0:
        return float("nan")
    min_value = float("inf")
    for value in filtered_values:
        if value < min_value:
            min_value = value
    return min_value


def values_max(values: Sequence[Any]) -> float:
    filtered_values = [value for value in values if not pd.isna(value)]
    if len(filtered_values) == 0:
        return float("nan")
    max_value = float("-inf")
    for value in filtered_values:
        if value > max_value:
            max_value = value
    return max_value


def percentile(values: Sequence[Any], percentage: int) -> float:
    filtered_values = [value for value in values if not pd.isna(value)]
    if len(filtered_values) == 0:
        return float("nan")
    sorted_values = sorted(filtered_values)
    position = (percentage / 100) * len(sorted_values)
    if position == 0:
        return sorted_values[0]
    elif position == len(sorted_values):
        return sorted_values[-1]
    if position.is_integer():
        return sorted_values[int(position) - 1]
    lower_index = int(position)
    upper_index = lower_index + 1
    fractional_part = position - lower_index
    return (sorted_values[lower_index] * (1 - fractional_part)) + (
        sorted_values[upper_index] * fractional_part
    )


def describe(dataframe: pd.DataFrame):
    table = PrettyTable()
    table.field_names = ["", *dataframe.columns]
    table.add_row(
        [
            "Count",
            *(
                round(count(dataframe[column].to_list()), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "Mean",
            *(
                round(mean(dataframe[column].to_list()), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "Std",
            *(
                round(std(dataframe[column].to_list()), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "Min",
            *(
                round(values_min(dataframe[column].to_list()), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "25%",
            *(
                round(percentile(dataframe[column].to_list(), 25), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "50%",
            *(
                round(percentile(dataframe[column].to_list(), 50), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "75%",
            *(
                round(percentile(dataframe[column].to_list(), 75), 5)
                for column in dataframe.columns
            ),
        ]
    )
    table.add_row(
        [
            "Max",
            *(
                round(values_max(dataframe[column].to_list()), 5)
                for column in dataframe.columns
            ),
        ]
    )
    print(table)


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)

    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def filter_non_numerical_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.select_dtypes(include=[np.number])


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
    df = filter_non_numerical_columns(df)
    describe(df)


if __name__ == "__main__":
    main()
