#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
from math import log
import warnings
warnings.filterwarnings("ignore")




def calc_entropy(total_num, count_dict):
    """
    计算信息熵
    :param total_num: 总样本数, 例如总的样本数是 14
    :param count_dict: 每类样本及其对应数目的字典，例如：{'前往游乐场': 9, '不前往游乐场': 5}
    :return: 信息熵
    """
    
    #初始化 ent 为 0
    ent = 0
    # 对于每一个类别
    for n in count_dict.values():
        # 如果属于该类别的样本数大于 0
        if n > 0:
            # 计算概率
            p = n / total_num
            # 计算信息熵
            ent += - p * log(p, 2)
    # 返回信息熵，精确到小数点后 3 位
    return round(ent, 3)


def decision_2():
    df = pd.read_csv('datasets.csv')
    # 样本总数
    total_num = 14
    # 每类样本的数量
    count_dict = {'去游乐场': 9, '不去游乐场': 5}
    # 例如：按是否前往游乐场==0 进行筛选
    df[df['是否前往游乐场']== 0 ]

    # 总样本数
    total_num = df.shape[0]

    # 前往游乐场的样本数目
    num_go_to_play = df[df['是否前往游乐场']== 1].shape[0]

    # 不前往游乐场的样本数目
    num_not_go_to_play = df[df['是否前往游乐场']== 0 ].shape[0]

    # 每类样本及其对应数目的字典
    count_dict = {'前往': num_go_to_play,
                  '不前往': num_not_go_to_play}

    # 计算信息熵
    entropy = calc_entropy(total_num, count_dict)

    # 筛选出 天气为晴并且去游乐场的样本数据
    df[(df['天气']=='晴') & (df['是否前往游乐场']== 1 )]
    # 天气为晴的总天数
    total_num_sun = df[df['天气']=='晴'].shape[0]

    # 前往游乐场的样本数目
    num_go_to_play = df[(df['天气']=='晴') & (df['是否前往游乐场']== 1 )].shape[0]

    # 不前往游乐场的样本数目
    num_not_go_to_play = df[(df['天气']=='晴') & (df['是否前往游乐场']== 0 )].shape[0]

    # 天气为晴时，去游乐场和不去游乐场的人数
    count_dict_sun = {'前往': num_go_to_play ,
                      '不前往': num_not_go_to_play}

    # 计算天气-晴 的信息熵
    ent_sun = calc_entropy(total_num_sun, count_dict_sun)

    # 天气为多云的总天数
    total_num_cloud = df[df['天气']=='多云'].shape[0]

    # 天气为多云时，去游乐场和不去游乐场的人数
    count_dict_cloud = {'前往':df[(df['天气']=='多云') & (df['是否前往游乐场']== 1 )].shape[0],
                        '不前往':df[(df['天气']=='多云') & (df['是否前往游乐场']== 0 )].shape[0]}
    # 计算天气-多云 的信息熵
    ent_cloud = calc_entropy(total_num_cloud, count_dict_cloud)

    # 天气为雨的总天数
    total_num_rain = df[df['天气']=='雨'].shape[0]

    # 天气为雨时，去游乐场和不去游乐场的人数
    count_dict_rain = {'前往':df[(df['天气']=='雨') & (df['是否前往游乐场']== 1 )].shape[0],
                  '不前往':df[(df['天气']=='雨') & (df['是否前往游乐场']== 0 )].shape[0]}

    # 计算天气-雨 的信息熵
    ent_rain = calc_entropy(total_num_rain, count_dict_rain)
    
    return df, entropy, total_num_sun, total_num, ent_sun, total_num_cloud, ent_cloud, total_num_rain, ent_rain

