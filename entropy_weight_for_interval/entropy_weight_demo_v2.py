# _*_coding:utf-8 _*_
# @Time     :2023/12/8 18:11
# @Author   : anliu
# @File     :entropy_weight_for_refund_rate.py
# @Theme    :对于正向指标和负向指标（越大越好的指标和越小越好的指标），可以分别进行不同的处理
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
        return (df[pos_col_list] - df[pos_col_list].min())/(df[pos_col_list].max()-df[pos_col_list].min())
    elif pos_col_list == None and neg_col_list != None:
        return (df[neg_col_list].max - df[neg_col_list])/(df[neg_col_list].max()-df[neg_col_list].min())
    else:
        a = (df[pos_col_list] - df[pos_col_list].min())/(df[pos_col_list].max()-df[pos_col_list].min())
        b = (df[neg_col_list].max() - df[neg_col_list])/(df[neg_col_list].max()-df[neg_col_list].min())
        return pd.concat([a, b], axis=1)

def get_entropy_weight(df):
    """
    :param df: 预处理好的数据
    :return: 输出权重-数组。
    """
    K = 1/np.log(len(df))
    p = df/np.sum(df)
    e = -K*np.sum(p*np.log(p))
    d = 1-e
    w = d/d.sum()
    return w

if __name__ == '__main__':
    # 1. 初始数据 假设指标4是负向指标，其余三个为正向指标
    df1 = pd.DataFrame({'指标1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        '指标2': [2, 4, 6, 8, 10, 2, 4, 6, 8, 10],
                        '指标3': [1, 2, 1, 3, 2, 1, 3, 2, 3, 1],
                        '指标4': [3, 1, 2, 3, 5, 8, 7, 8, 8, 9]
                        })
    df_standardization = standardization(df1, ['指标1', '指标2', '指标3'], ['指标4'])
    print('标准化处理:\n', df_standardization)
    print(df_standardization)
    w = get_entropy_weight(df_standardization)
    print('指标权重数组:\n', w)
    df_score = df_standardization*w
    df_score['score'] = df_score.sum(axis=1)
    print('综合得分:\n', df_score)
