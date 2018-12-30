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

    df = pd.DataFrame(np.zeros((x.shape[0], 1)))
    df.columns = ["to_delete"]
    name_list = []

    for comb in list(itertools.combinations(x.columns.values, 2)):
        plus_df = x.loc[:, comb].sum(axis=1)
        name = str(comb[0])+'+'+str(comb[1])
        name_list.append(name)
        df = pd.concat((df, plus_df), axis=1)

    df = df.drop("to_delete", axis=1)
    df.columns = name_list

    return df

