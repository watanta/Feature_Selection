"""
Permutation Testを行い、重要度が高い説明変数を抽出する。
"""

import eli5
from eli5.sklearn import PermutationImportance
import pandas as pd


def get_permutation_score(x, y, estimator, cv=5, random_state=1):
    """

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        estimator (object): 学習器
        cv (int, default=5): cross-validationの分割数
        random_state (int, default=5): ランダムシード

    Returns:
        (DataFrame): 各説明変数のpermutation scoreの平均と標準偏差

    """

    perm = PermutationImportance(estimator, random_state=random_state, cv=cv).fit(x.values, y.values.flatten())
    perm_df = pd.DataFrame(perm.feature_importances_.reshape(1, -1))
    perm_df.index = ['perm_importance_mean']
    perm_df.columns = x.columns
    perm_df.loc['per_importance_std', :] = perm.feature_importances_std_.reshape(1, -1)

    return perm_df


def perm_selection_rank(x, y, estimator, rank, cv=5, random_state=1):
    """permutation_scoreの平均値によって特徴量選択をする

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        rank: この順位以下の説明変数は削除される
        estimator (object): 学習器
        cv (int, default=5): cross-validationの分割数
        random_state (int, default=5): ランダムシード

    Returns:
        DataFrame: 選択された特徴量

    """

    assert rank < x.shape[1], "rankが説明変数の数より大きいです"

    perm_df = x.copy()
    perm = get_permutation_score(x, y, estimator=estimator, cv=cv, random_state=random_state)
    perm_df.loc["perm_importance_mean", :] = perm.loc["perm_importance_mean",:].values
    perm_df = perm_df.sort_values("perm_importance_mean", ascending=False, axis=1)
    selected_columns = perm_df.columns.values[:rank]
    perm_df = perm_df.loc[:, selected_columns]

    return perm_df


def perm_selection_th(x, y, estimator, th, cv=5, random_state=1):
    """permutation_scoreの平均値によって特徴量選択をする

    Args:
        x (DataFrame): 説明変数
        y (DataFrame): 目的変数
        th: この閾値以下のpermutation socreの説明変数は削除される
        estimator (object): 学習器
        cv (int, default=5): cross-validationの分割数
        random_state (int, default=5): ランダムシード

    Returns:
        DataFrame: 選択された特徴量

    """

    perm_df = x.copy()
    perm = get_permutation_score(x, y, estimator=estimator, cv=cv, random_state=random_state)
    perm_df.loc["perm_importance_mean", :] = perm.loc["perm_importance_mean", :].values
    perm_df = perm_df.sort_values("perm_importance_mean", ascending=False, axis=1)
    perm_df = perm_df.loc[:, perm_df.loc["perm_importance_mean", :] > th]

    return perm_df

