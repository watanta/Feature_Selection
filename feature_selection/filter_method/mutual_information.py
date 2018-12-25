"""相互情報量に基づいて特徴量選択を行うモジュール"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def get_mutual_information(x, y):
    """
    各説明変数と目的変数の相互情報量を計算する

    :param x: 説明変数のDataFrame
    :param y: 目的変数のDataFrame
    :return: 各説明変数と目的変数の相互情報量
    """

    mi_list = []
    for key, column in x.iteritems():
        mi = mutual_info_regression(column.values.reshape(-1,1), y.values.reshape(-1,))
        mi_list.append(mi)

    mi_df = pd.DataFrame(np.array(mi_list).reshape(1, -1))
    mi_df.columns = x.columns
    mi_df.index = ["mutation_information"]

    return mi_df

