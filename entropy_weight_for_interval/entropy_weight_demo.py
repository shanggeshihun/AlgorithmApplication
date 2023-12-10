# _*_coding:utf-8 _*_
# @Time     :2023/12/8 18:11
# @Author   : anliu
# @File     :entropy_weight_for_refund_rate.py
# @Theme    :PyCharm

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


def get_score(wi_list, df):
    """
    :param wi_list: 权重系数列表
    :param df：评价指标数据框
    :return:返回得分
    """

    #  将权重转换为矩阵
    cof_var = np.mat(wi_list)

    #  将数据框转换为矩阵
    context_train_df = np.mat(df)

    #  权重跟自变量相乘
    last_hot_matrix = context_train_df * cof_var.T
    last_hot_matrix = pd.DataFrame(last_hot_matrix)

    #  累加求和得到总分
    last_hot_score = list(last_hot_matrix.apply(sum))

    #  max-min 归一化

    # last_hot_score_autoNorm = autoNorm(last_hot_score)

    # 值映射成分数（0-100分）

    # last_hot_score_result = [i * 100 for i in last_hot_score_autoNorm]

    return last_hot_score


def get_entropy_weight(df):
    """
    :param df: 评价指标数据框
    :return: 各指标权重列表
    """
    # 数据标准化
    df = (df - df.min()) / (df.max() - df.min())
    m, n = df.shape
    # 将DataFrame格式转化为matrix格式
    df = df.as_matrix(columns=None)
    k = 1 / np.log(m)
    yij = df.sum(axis=0)
    # 第二步，计算pij
    pij = df / yij

    p_log = pij * np.log(pij)
    p_log = np.nan_to_num(p_log)

    # 计算每种指标的信息熵
    ej = -k * (p_log.sum(axis=0))
    # 计算每种指标的权重
    wi = (1 - ej) / np.sum(1 - ej)
    wi_list = list(wi)

    return wi_list


if __name__ == '__main__':
    ##
    # df0 = pd.read_excel("C:\\Users\\Oreo\\Desktop\\p_log2.xlsx", encoding='utf8')
    df0 = pd.DataFrame(
        {
            '科室': ['a', 'b', 'c', 'd', 'e', 'f'],
            'x1': [0, 0, 0, 1, 2, 3],
            'x2': [4, 8, 5, 0, 0, 0]
         }
    )
    df = df0.iloc[:, 1:10]
    print(type(df))

    mm = df
    wi_list = get_entropy_weight(df)
    score_list = get_score(mm, wi_list)
    mm['score'] = score_list
    mm['科室'] = df0['科室']
    # 然后对数据框按得分从大到小排序
    result = mm.sort_values(by='score', axis=0, ascending=False)
    result['rank'] = range(1, len(result) + 1)
