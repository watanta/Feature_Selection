"""各列の組から新しい特徴量を作成する"""

import pandas as pd
import numpy as np
import itertools


def plus(x):
    """

    Args:
        x (DataFrame): 説明変数

    Returns:
        DataFrame: xのすべての列の要素が2の組それぞれの和

    """

    print("plus...")

    df = pd.DataFrame(np.zeros((x.shape[0], 1)))
    df.columns = ["to_delete"]
    name_list = []
    i = 0

    for comb in list(itertools.combinations(x.columns.values, 2)):
        print(i/len(list(itertools.combinations(x.columns.values, 2))))
        plus_df = x.loc[:, comb].sum(axis=1)
        name = str(comb[0])+'+'+str(comb[1])
        name_list.append(name)
        df = pd.concat((df, plus_df), axis=1)

        i += 1

    df = df.drop("to_delete", axis=1)
    df.columns = name_list

    return df


def minus(x):
    """

    Args:
        x (DataFrame): 説明変数

    Returns:
        DataFrame: xのすべての列の要素が2の組それぞれの差

    """
    print("minus...")

    df = pd.DataFrame(np.zeros((x.shape[0], 1)))
    df.columns = ["to_delete"]
    name_list = []
    i = 0

    for comb in list(itertools.combinations(x.columns.values, 2)):
        print(i / len(list(itertools.combinations(x.columns.values, 2))))
        x1 = x.loc[:, comb[0]]
        x2 = x.loc[:, comb[1]]
        minus_df = x1 - x2
        name = str(comb[0])+'-'+str(comb[1])
        name_list.append(name)
        df = pd.concat((df, minus_df), axis=1)

        i += 1

    df = df.drop("to_delete", axis=1)
    df.columns = name_list

    return df


def times(x):
    """

    Args:
        x (DataFrame): 説明変数

    Returns:
        DataFrame: xのすべての列の要素が2の組それぞれの積

    """
    print("times...")

    df = pd.DataFrame(np.zeros((x.shape[0], 1)))
    df.columns = ["to_delete"]
    name_list = []
    i = 0

    for comb in list(itertools.combinations(x.columns.values, 2)):
        print(i / len(list(itertools.combinations(x.columns.values, 2))))
        x1 = x.loc[:, comb[0]]
        x2 = x.loc[:, comb[1]]
        times_df = x1 * x2
        name = str(comb[0])+'*'+str(comb[1])
        name_list.append(name)
        df = pd.concat((df, times_df), axis=1)

        i += 1

    df = df.drop("to_delete", axis=1)
    df.columns = name_list

    return df


def div(x):
    """

    Args:
        x (DataFrame): 説明変数

    Returns:
        DataFrame: xのすべての列の要素が2の組それぞれの商

    """

    print("div...")

    df = pd.DataFrame(np.zeros((x.shape[0], 1)))
    df.columns = ["to_delete"]
    name_list = []
    i = 0

    for comb in list(itertools.combinations(x.columns.values, 2)):
        print(i / len(list(itertools.combinations(x.columns.values, 2))))
        x1 = x.loc[:, comb[0]]
        x2 = x.loc[:, comb[1]]
        div_df = x1 * x2
        name = str(comb[0])+'/'+str(comb[1])
        name_list.append(name)
        df = pd.concat((df, div_df), axis=1)
        i += 1

    df = df.drop("to_delete", axis=1)
    df.columns = name_list

    return df


def all_arithmetic(x):
    """各組に対してすべての四則演算をする

    Args:
        x (DataFrame): 説明変数

    Returns:
        DataFrame: xのすべての列の要素が2の組それぞれの和、差、積、商


    """

    plus_df = plus(x)
    minus_df = minus(x)
    times_df = times(x)
    div_df = div(x)

    df = pd.concat((x, plus_df), axis=1)
    df = pd.concat((df, minus_df), axis=1)
    df = pd.concat((df, times_df), axis=1)
    df = pd.concat((df, div_df), axis=1)

    return df

