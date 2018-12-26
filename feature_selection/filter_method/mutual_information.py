"""相互情報量に基づいて特徴量選択を行うモジュール"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def get_mutual_information(x, y):
    """各説明変数と目的変数の相互情報量を取得する。

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数

    Returns:
        DataFrame: 各説明変数と目的変数の相互情報量

    """

    mi_list = []
    for key, column in x.iteritems():
        mi = mutual_info_regression(column.values.reshape(-1,1), y.values.reshape(-1,))
        mi_list.append(mi)

    mi_df = pd.DataFrame(np.array(mi_list).reshape(1, -1))
    mi_df.columns = x.columns
    mi_df.index = ["mutation_information"]

    return mi_df


def mi_selection_th(x, y, th):
    """
    相互情報量によって特徴量選択をする

    :param x: 説明変数のDataFrame
    :param y: 目的変数のDataFrame
    :param th: 相互情報量の閾値　この値以下の説明変数は削除される
    :return: 選択された特徴量のDataFrame
    """

    assert 0 < th, 'thは0<thにしてください'

    mi_df = x.copy()
    mi = get_mutual_information(x, y)
    mi_df.loc["mi", :] = mi.values
    mi_df = mi_df.sort_values('mi', ascending=False, axis=1)
    mi_df = mi_df.loc[:, mi_df.loc["mi", :] > th]

    return mi_df


def mi_selection_rank(x, y, rank):
    """
    各説明変数と目的変数の相関係数を計算して特徴量選択を行う

    :param dataframe x: 入力データ
    :param dataframe y:  目的変数のdataframe
    :param real rank: 選択する特徴量の順位の閾値。この値以下の順位の説明変数は削除される。
    :return:  dfのうちの選ばれた説明変数

    """

    assert rank < x.shape[1], "rankが説明変数の数より大きいです"

    mi_df = x.copy()
    mi = get_mutual_information(x, y)
    mi_df.loc["mi", :] = mi.values
    mi_df = mi_df.sort_values('mi', ascending=False, axis=1)
    selected_columns = mi_df.columns.values[:rank]
    mi_df = mi_df.loc[:, selected_columns]

    return mi_df
