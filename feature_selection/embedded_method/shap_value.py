"""
SHAP valueに基づいて特徴量選択を行う
"""

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


def get_shap_value(x, y, estimator, ex_type="tree", cv=5):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        ex_type (str): shapのexplainerの種類 treeかkernel
        cv (int): xの分割数

    Returns:
        DataFrame: 各説明変数のshap_valueの絶対値の平均.
                    汎化したshap valueを見たいので、cvしてvalidationのshap valuesを平均する
    """

    ms = MinMaxScaler()
    data_norm = ms.fit_transform(x)
    kf = KFold(n_splits=cv)
    cv_list = []
    cv_index = []
    i = 0

    for train_index, valid_index in kf.split(data_norm):

        i +=1

        train_x = x.loc[train_index, :]
        train_y = y.loc[train_index, :]
        valid_x = x.loc[valid_index, :]

        model = estimator.fit(train_x, train_y)
        if ex_type == "tree":
            explainer = shap.TreeExplainer(model)
        if ex_type == "kernel":
            explainer = shap.KernelExplainer(model)

        shap_values = explainer.shap_values(valid_x)
        shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

        cv_list.append(shap_values_abs_mean)
        cv_index.append("cv"+str(i))

    shap_cv = pd.DataFrame(np.array(cv_list).reshape(cv, x.shape[1]))
    shap_cv.columns = x.columns
    shap_cv.index = cv_index
    shap_cv.loc["shap_mean"] = shap_cv.mean(axis=0)

    return shap_cv


def get_shap_value_moment(x, y, estimator, ex_type="tree", cv=5):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        ex_type (str): shapのexplainerの種類 treeかkernel
        cv (int): xの分割数

    Returns:
        DataFrame: 各説明変数のshap valueの0まわりのモーメント (位置がshap valueで質量がその説明変数の値)
                    汎化したshap valueを見たいので、cvしてvalidationのshap valuesを平均する

    """

    kf = KFold(n_splits=cv)
    cv_list = []
    cv_index = []
    i = 0

    for train_index, valid_index in kf.split(x):

        i +=1

        train_x = x.loc[train_index, :]
        train_y = y.loc[train_index, :]
        valid_x = x.loc[valid_index, :]

        model = estimator.fit(train_x, train_y)
        if ex_type == "tree":
            explainer = shap.TreeExplainer(model)
        if ex_type == "kernel":
            explainer = shap.KernelExplainer(model)

        shap_values = explainer.shap_values(valid_x)
        shap_values_moment = np.abs((valid_x * np.abs(shap_values)).mean(axis=0))

        cv_list.append(shap_values_moment)
        cv_index.append("cv"+str(i))

    shap_cv = pd.DataFrame(np.array(cv_list).reshape(cv, x.shape[1]))
    shap_cv.columns = x.columns
    shap_cv.index = cv_index
    shap_cv.loc["shap_moment_mean"] = shap_cv.mean(axis=0)

    return shap_cv



def shap_value_selection_rank(x, y, estimator, rank, cv=5, ex_type='tree'):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        rank (int): この順位未満の説明変数は削除される
        cv (int): xの分割数
        ex_type: shapのexplainerの種類 treeかkernel

    Returns:
        DataFrame: 選択された説明変数

    """

    assert rank < x.shape[1], "rankが説明変数の数より大きいです"

    shap_df = x.copy()
    shap = get_shap_value(x, y, estimator=estimator, cv=cv)
    shap_df.loc["shap_mean", :] = shap.loc["shap_mean", :].values
    shap_df = shap_df.sort_values("shap_mean", ascending=False, axis=1)
    selected_columns = shap_df.columns.values[:rank]
    shap_df = shap_df.loc[:, selected_columns]

    return shap_df


def shap_value_selection_th(x, y, estimator, th, cv=5, ex_type='tree'):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        th (real): この閾値未満の説明変数は削除される
        cv (int): xの分割数
        ex_type: shapのexplainerの種類 treeかkernel

    Returns:
        DataFrame: 選択された説明変数

    """

    shap_df = x.copy()
    shap = get_shap_value(x, y, estimator=estimator, cv=cv)
    shap_df.loc["shap_mean", :] = shap.loc["shap_mean", :].values
    shap_df = shap_df.sort_values("shap_mean", ascending=False, axis=1)
    shap_df = shap_df.loc[:, shap_df.loc["shap_mean", :] > th]

    return shap_df

def shap_value_moment_selection_rank(x, y, estimator, rank, cv=5, ex_type='tree'):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        rank (int): この順位未満の説明変数は削除される
        cv (int): xの分割数
        ex_type: shapのexplainerの種類 treeかkernel

    Returns:
        DataFrame: 選択された説明変数

    """

    assert rank < x.shape[1], "rankが説明変数の数より大きいです"

    shap_df = x.copy()
    shap = get_shap_value_moment(x, y, estimator=estimator, cv=cv)
    shap_df.loc["shap_moment_mean", :] = shap.loc["shap_moment_mean", :].values
    shap_df = shap_df.sort_values("shap_moment_mean", ascending=False, axis=1)
    selected_columns = shap_df.columns.values[:rank]
    shap_df = shap_df.loc[:, selected_columns]

    return shap_df


def shap_value_moment_selection_th(x, y, estimator, th, cv=5, ex_type='tree'):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        th (real): この閾値未満の説明変数は削除される
        cv (int): xの分割数
        ex_type: shapのexplainerの種類 treeかkernel

    Returns:
        DataFrame: 選択された説明変数

    """

    shap_df = x.copy()
    shap = get_shap_value_moment(x, y, estimator=estimator, cv=cv)
    shap_df.loc["shap_moment_mean", :] = shap.loc["shap_moment_mean", :].values
    shap_df = shap_df.sort_values("shap_moment_mean", ascending=False, axis=1)
    shap_df = shap_df.loc[:, shap_df.loc["shap_moment_mean", :] > th]

    return shap_df

