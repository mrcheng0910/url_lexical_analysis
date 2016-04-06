#!/usr/bin/python
# encoding:utf-8
"""
比较五种分类算法的正确率
作者：程亚楠
创建时间：2016.4.6
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'

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
    plt.plot(x, dt, label='DT', color='r', linewidth=2)
    plt.plot(x, svm, label='SVM', color='b', linewidth=2)
    plt.plot(x, logistic, label='Logistic', color='k', linewidth=2)
    plt.plot(x, kneighbors, label='KNeighbors', color='c', linewidth=2)
    plt.plot(x, gaussianNB, label='GaussianNB', color='m', linewidth=2)
    plt.xlabel(u'数据集网址数量(个)')
    plt.ylabel(u'预测正确率')
    plt.title(u'五种分类算法正确率比较')
    plt.legend()
    plt.show()

def get_xyaxis_data():
    """
    获得测试数据个数(横坐标)和各个算法的正确率(纵坐标)
    :return: x和ys
    """
    x = [200, 400, 800,1500,2000]

    dt = [0.92,0.93,0.94]
    svm = [0.83,0.84,0.87]
    logistic = [0.82,0.83,0.94]
    kneighbors = [0.72,0.75,0.76]
    gaussianNB = [0.6,0.65,0.65]
    return x, dt,svm,logistic,kneighbors,gaussianNB


def get_dt_yaxis():
    """
    获取决策树算法的准确率
    :return:
    """
    print "nihao"


def main():
    """
    主函数
    """
    x, dt,svm,logistic,kneighbors,gaussianNB = get_xyaxis_data()
    draw_comparing_graphy(x, dt,svm,logistic,kneighbors,gaussianNB)

if __name__ == '__main__':
    main()
