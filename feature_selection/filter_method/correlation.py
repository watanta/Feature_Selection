"""相関係数に基づいて特徴量選択を行うモジュール"""

import numpy as np
import pandas as pd


def correlation_selection_th(x, y, th):
    """
    各説明変数と目的変数の相関係数を計算して特徴量選択を行う

    :param dataframe x: 入力データ
    :param dataframe y:  目的変数のdataframe
    :param real th: 相関係数の絶対値の閾値。この値以下の説明変数は削除される。
    :return:  dfのうちの選ばれた説明変数


    """

    assert 0 < th < 1, 'thは0<th<1でなければならない'

    corr_df = x.copy()
    corr = get_abs_correlations(x, y)
    corr_df.loc["corr", :] = corr.values
    corr_df = corr_df.sort_values('corr', ascending=False, axis=1)
    corr_df = corr_df.loc[:, corr_df.loc["corr", :] > th]

    return corr_df


def correlation_selection_rank(x, y, rank):
    """
    各説明変数と目的変数の相関係数を計算して特徴量選択を行う

    :param dataframe x: 入力データ
    :param dataframe y:  目的変数のdataframe
    :param real rank: 選択する特徴量の順位の閾値。この値以下の順位の説明変数は削除される。
    :return:  dfのうちの選ばれた説明変数


    """

    corr_df = x.copy()
    corr = get_abs_correlations(x, y)
    corr_df.loc["corr", :] = corr.values
    corr_df = corr_df.sort_values('corr', ascending=False, axis=1)
    selected_columns = corr_df.columns.values[:rank]
    corr_df = corr_df.loc[:, selected_columns]
    # corr_df = corr_df.drop('corr', axis=0)

    return corr_df


def get_correlations(x, y):
    """
    各説明変数と目的変数との相関係数を計算する

    :param  dataframe x: 説明変数のdataframe
    :param dataframe y:  目的変数のdataframe
    :return: 各説明変数と目的変数の相関係数


    """

    corr_list = []
    for key, column in x.iteritems():
        column_df = pd.DataFrame(column)

        corr = np.corrcoef(column_df.values.reshape((1, -1)), y.values.reshape((1, -1)))[0, 1]
        corr_list.append(corr)

    corr_df = pd.DataFrame(np.array(corr_list).reshape(1, -1))
    corr_df.columns = x.columns
    corr_df.index = ["correlation"]

    return corr_df


def get_abs_correlations(x, y):
    """
    各説明変数と目的変数との相関係数を計算する

    :param  dataframe x: 説明変数のdataframe
    :param dataframe y: 目的変数のdataframe
    :return: 各説明変数と目的変数の相関係数の絶対値

    """

    corr_list = []
    for key, column in x.iteritems():
        column_df = pd.DataFrame(column)

        corr = np.corrcoef(column_df.values.reshape((1, -1)), y.values.reshape((1, -1)))[0, 1]
        corr_list.append(corr)

    corr_df = pd.DataFrame(np.abs(np.array(corr_list)).reshape(1, -1))
    corr_df.columns = x.columns
    corr_df.index = ["correlation"]

    return corr_df

