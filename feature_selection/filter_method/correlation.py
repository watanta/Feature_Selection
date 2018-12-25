"""相関係数に基づいて特徴量選択を行うモジュール"""

import numpy as np
import pandas as pd


def correlation_selection(df, target, th):
    """
    各説明変数と目的変数の相関係数を計算して特徴量選択を行う

    :param dataframe df: 入力データ
    :param dataframe target:  目的変数のdataframe
    :param real th: 相関係数の絶対値の閾値。この値以下の説明変数は削除される。
    :return:  dfのうちの選ばれた説明変数


    """

    corr_df = df.copy()

    for key, column in df.iteritems():
        column_df = pd.DataFrame(column)

        corr = np.abs(np.corrcoef(column_df.values.reshape((1, -1)), target.values.reshape((1, -1)))[0, 1])

        if corr < th:
            corr_df = corr_df.drop(key, axis=1)

    return corr_df


def get_correlations(df, target):
    """
    各説明変数と目的変数との相関係数を計算する

    :param  dataframe df: 説明変数のdataframe
    :param dataframe target:  目的変数のdataframe
    :return: 各説明変数と目的変数の相関係数


    """

    corr_list = []
    for key, column in df.iteritems():
        column_df = pd.DataFrame(column)

        corr = np.corrcoef(column_df.values.reshape((1, -1)), target.values.reshape((1, -1)))[0, 1]
        corr_list.append(corr)

    corr_df = pd.DataFrame(np.array(corr_list).reshape(1, -1))
    corr_df.columns = df.columns
    corr_df.index = ["correlation"]

    return corr_df


def get_abs_correlations(df, target):
    """
    各説明変数と目的変数との相関係数を計算する

    :param  dataframe df: 説明変数のdataframe
    :param dataframe target: 目的変数のdataframe
    :return: 各説明変数と目的変数の相関係数の絶対値

    """

    corr_list = []
    for key, column in df.iteritems():
        column_df = pd.DataFrame(column)

        corr = np.corrcoef(column_df.values.reshape((1, -1)), target.values.reshape((1, -1)))[0, 1]
        corr_list.append(corr)

    corr_df = pd.DataFrame(np.abs(np.array(corr_list)).reshape(1, -1))
    corr_df.columns = df.columns
    corr_df.index = ["correlation"]

    return corr_df

