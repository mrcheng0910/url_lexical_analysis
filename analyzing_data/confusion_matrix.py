# encoding:utf-8
"""
各个算法的混淆矩阵图像绘制
作者：程亚楠
创建时间：2016.4.7
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datamining import extract_key_feature_data, train_test_split_data, svm_data, k_neighbor_data, gausssian_data
from datamining import logistic_data

rcParams['font.family'] = 'SimHei'  # 支持中文字体


def plot_confusion_matrix(cm,title=u'SVM分类恶意域名混淆矩阵图', cmap=plt.cm.Blues,normalized=False):
    """
    绘制矩阵图
    :param cm:
    :param title:
    :param cmap:
    """
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [u'恶意网址',u'良性网址'])
    plt.yticks(tick_marks, [u'恶意网址',u'良性网址'], rotation=90)
    # plt.tight_layout()
    plt.ylabel(u'真实结果')
    plt.xlabel(u'预测结果')
    # 设置数据标签
    if normalized:  # 保留两位小数
        tp = round(cm[0][0],2)  # good-->good
        fp = round(cm[0][1],2)  # good-->bad
        fn = round(cm[1][0],2)  # bad-->good
        tn = round(cm[1][1],2)  # bad-->bad
    else:
        tp = cm[0][0]  # good-->good
        fp = cm[0][1]  # good-->bad
        fn = cm[1][0]  # bad-->good
        tn = cm[1][1]  # bad-->bad
    tp_label = plt.text(0.30, 0.40, tp)
    plt.setp(tp_label, color='w')
    fp_label = plt.text(0.57, 0.40, fp)
    plt.setp(fp_label, color='k')
    fn_label = plt.text(0.30, 0.65, fn)
    plt.setp(fn_label, color='k')
    tn_label = plt.text(0.57, 0.65, tn)
    plt.setp(tn_label, color='w')


def compute_confusion_matrix(cm,title='SVM',normalized = False):
    """
    计算矩阵，并且绘制矩阵图
    :param cm: 矩阵数据
    :param title: 标题
    :param normalized: 是否归一化
    """
    plt.figure()
    if normalized == True:
        title= u'%s分类恶意域名混淆矩阵图(百分比)' % title
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized, title=title,normalized=True)
    else:
        title= u'%s分类恶意域名混淆矩阵图(数量)' % title
        plot_confusion_matrix(cm,title=title)

    plt.show()


def get_df():
    """
    获取训练数据集
    :return: 关键特征数据集，结果数据集
    """
    df, y, sub_columns = extract_key_feature_data()
    x_train, x_test, y_train, y_test = train_test_split_data(df,y,train_size=0.90)
    return x_train, y_train


def get_cm():
    """
    获取confusion_matrix数据
    :return: 矩阵数据
    """
    x,y = get_df()
    _,log_cm = logistic_data(x,y)
    _,svm_cm = svm_data(x,y)
    _,kn_cm = k_neighbor_data(x,y)
    _,g_cm = gausssian_data(x,y)
    return g_cm,log_cm,kn_cm,svm_cm


def main():
    """
    主函数
    """
    g_cm,log_cm,kn_cm,svm_cm = get_cm()
    compute_confusion_matrix(g_cm,'G', True)
    compute_confusion_matrix(log_cm,'log', True)
    compute_confusion_matrix(kn_cm,'kn', True)
    compute_confusion_matrix(svm_cm,'svm', True)


if __name__ == '__main__':
    main()