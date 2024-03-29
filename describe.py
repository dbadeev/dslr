"""
Describe program that shows key statistic metrics for a dataset
"""
import argparse
import pandas as pd
import numpy as np
from maths import count_v, mean, std, min_val, max_val, quartile_25, quartile_50, quartile_75
from maths import var, sum_of_squares, skewness, kurtosis_42, mode_42
from messages import ProgramError


def get_dataframe(datafile: str):
    """
    Reads hogwarts dataset to a dataframe

    :param datafile:
    :return: dataframe with hogwarts dataset
    """
    try:
        # pylint: disable=E1136
        df_data = pd.read_csv(datafile, index_col='Index')
        if 'Best Hand' in df_data.columns:
            df_data['Best Hand'].replace({'Left': 0, 'Right': 1}, inplace=True)
        return df_data
    except (Exception, FileNotFoundError) as error:
        raise ProgramError(error) from error


def describe(df_data, extended=False):
    """
    Calculates statistic metrics and displays them (like pd.describe())

    :param extended: option for showing extra metrics
    :param df_data: pd.dataframe with a data set
    """
    try:
        numeric_columns = list(df_data.select_dtypes(include=[np.number]).columns.values)
        funcs_map = {'count': count_v,
                     'mean': mean,
                     'std': std,
                     'min': min_val,
                     '25%': quartile_25,
                     '50%': quartile_50,
                     '75%': quartile_75,
                     'max': max_val}
        if extended:
            funcs_map['variation'] = var
            funcs_map['sum of squares'] = sum_of_squares
            funcs_map['skewness'] = skewness
            funcs_map['kurtosis'] = kurtosis_42
            funcs_map['mode'] = mode_42
        metrics = list(funcs_map)
        df_describe = pd.DataFrame(columns=numeric_columns, index=metrics)

        for hog_class in numeric_columns:
            for metric in metrics:
                df_describe.loc[metric][hog_class] = funcs_map[metric](df_data[hog_class].values)
        pd.set_option('display.max_columns', None)
        pd.options.display.float_format = '{:,.6f}'.format
        print(df_describe)
    except Exception as error:
        raise ProgramError(error) from error


def main():
    """
    Main program function that shows key statistic metrics for a dataset
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--extended', '-e',
                            action="store_true",
                            help='Additional statistic metrics')
        parser.add_argument('datafile',
                            type=str,
                            help='Path to dataset file')
        args = parser.parse_args()
        df_data = get_dataframe(args.datafile)
        describe(df_data, args.extended)
    except ProgramError as error:
        print(error)


if __name__ == '__main__':
    main()
