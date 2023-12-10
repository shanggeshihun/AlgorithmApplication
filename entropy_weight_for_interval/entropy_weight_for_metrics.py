# _*_coding:utf-8 _*_
# @Time     :2023/12/8 18:11
# @Author   : anliu
# @File     :entropy_weight_for_metrics.py
# @Theme    :对不同指标计算权重
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


def standardization(df, pos_col_list=None, neg_col_list=None):
    """
    :param df:目标数据
    :param pos_col_list: 需要处理的正向指标列名列表，类型为列表或None
    :param neg_col_list: 需要处理的负向指标列名列表，类型为列表或None
    :return: 输出处理结果
    """
    if pos_col_list == None and neg_col_list == None:
        return df
    elif pos_col_list != None and neg_col_list == None:
        return (df[pos_col_list] - df[pos_col_list].min()) / (df[pos_col_list].max() - df[pos_col_list].min())
    elif pos_col_list == None and neg_col_list != None:
        return (df[neg_col_list].max - df[neg_col_list]) / (df[neg_col_list].max() - df[neg_col_list].min())
    else:
        a = (df[pos_col_list] - df[pos_col_list].min()) / (df[pos_col_list].max() - df[pos_col_list].min())
        b = (df[neg_col_list].max() - df[neg_col_list]) / (df[neg_col_list].max() - df[neg_col_list].min())
        return pd.concat([a, b], axis=1)


def get_entropy_weight(df):
    """
    :param df: 预处理好的数据
    :return: 输出权重-数组。
    """
    K = 1 / np.log(len(df))
    p = df / np.sum(df)
    e = -K * np.sum(p * np.log(p))
    d = 1 - e
    w = d / d.sum()
    return w


if __name__ == '__main__':
    data0 = pd.read_excel(r'./data/parent_merchant_basedata_for_cluster.xlsx', sheet_name='Sheet2', encoding='utf8')

    pos_col_list = ['create_gap_days', 'last_7d_succ_orders', 'last_30d_succ_orders', 'trade_contribution', 'trade_median_amount', 'trade_amount', ]
    neg_col_list = ['kicked_status', 'refund_rate', 'chargeback_rate', 'warning_order_rate']

    cols = pos_col_list + neg_col_list
    df0 = data0[cols]

    df_standardization = standardization(df0, pos_col_list, neg_col_list)
    df_standardization.to_excel(r'./data/df_standardization.xlsx')
    sys.exit()
    w = get_entropy_weight(df_standardization)
    print('变量权重：\n', w)

    df_score = df_standardization * w
    df_score['score'] = df_score.sum(axis=1)

    df_score.to_excel(r'./data/entropy_weight_for_metrics.xlsx')
