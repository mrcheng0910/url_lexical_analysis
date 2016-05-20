#!/usr/bin/python
# encoding:utf-8
"""
比较五种分类算法的正确率
作者：程亚楠
创建时间：2016.4.6
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datamining import extract_key_feature_data
from datamining import train_test_split_data
from datamining import cal_decision_tree
from datamining import svm_data
from datamining import k_neighbor_data
from datamining import gausssian_data
from datamining import logistic_data
import  time

rcParams['font.family'] = 'SimHei'  # 支持中文字体
# 如果要保存为pdf格式，需要增加如下配置
#rcParams["pdf.fonttype"] = 42


def draw_comparing_graphy(x,dt,svm,logistic,kneighbors,gaussianNB):
    """
    绘制各个算法的曲线图
    :param x: 横坐标
    :param dt: 决策树
    :param svm: 支持向量机
    :param logistic: 逻辑回归
    :param kneighbors: k临接算法
    :param gaussianNB: 朴素贝叶斯算法
    """

    plt.plot(x, svm,'*-', label='SVM', color='b', linewidth=2)
    # plt.plot(x, dt, 'ko-',label='DT', color='r', linewidth=2)
    # plt.plot(x, logistic,'>-',label='Logistic', color='k', linewidth=2)
    # plt.plot(x, kneighbors,'v-', label='KNeighbors', color='c', linewidth=2)
    # plt.plot(x, gaussianNB, 'd-',label='NB', color='m', linewidth=2)
    plt.xlabel(u'网址数量(个)')
    plt.ylabel(u'正确率')
    # plt.title(u'3种分类算法正确率比较')
    # plt.legend(loc='center right')
    # plt.legend(loc=(0.7,0.6))
    plt.legend()
    plt.show()


def get_xy_data():
    """
    获得测试数据个数(横坐标)和各个算法的正确率(纵坐标)
    :return: x和ys
    """
    train_sizes = [0.5,0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.97]
    df,y,sub_columns = extract_key_feature_data()
    x = []
    dt = []
    svm = []
    logistic = []
    kneighbors = []
    gaussianNB = []
    for train_size in train_sizes:
        start = time.time()
        print start
        x_train,x_test,y_train,y_test = train_test_split_data(df, y, train_size=train_size)
        x.append(len(x_train))
        # dt_score, _, _ = cal_decision_tree(x_train,y_train,x_test,y_test)
        svm_score, _ = svm_data(x_train,y_train)
        # k_score, _ = k_neighbor_data(x_train,y_train)
        # g_score,_ = gausssian_data(x_train,y_train)
        # l_score,_ = logistic_data(x_train,y_train)
        end = time.time()
        print end
        print end-start
        # dt.append(dt_score)
        svm.append(svm_score)
        # logistic.append(l_score)
        # kneighbors.append(k_score)
        # gaussianNB.append(g_score)

    return np.array(x), np.array(dt), np.array(svm), np.array(logistic), np.array(kneighbors), np.array(gaussianNB)


def main():
    """
    主函数
    """
    x, dt, svm, logistic, kneighbors, gaussianNB = get_xy_data()
    draw_comparing_graphy(x, dt,svm,logistic,kneighbors,gaussianNB)


if __name__ == '__main__':
    main()
