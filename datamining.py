#!/usr/bin/python
# encoding:utf-8

"""
分析通过决策树的效果特征
"""
from data_base import MySQL
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

def fetch_data():
    """
    从数据库中获取数据基础数据
    :return: 返回基础数据
    """
    db = MySQL()
    sql = 'SELECT url_length,path_tokens,path_brand,domain_tokens,malicious FROM url_features'
    db.query(sql)
    urls = db.fetch_all_rows()
    db.close()
    return urls


def create_data_set():
    """
    根据获取的基础数据，建立对应的测试数据集合
    :return: 数据集合
    """
    urls = fetch_data()  # url基础数据

    url_length = []  # url的长度

    # url的path中相关信息
    path_count = []  # path中token数量
    path_total_length = []  # path的总长度
    path_avg_length = []  # path的平均长度
    path_max_length = []  # path的最大长度
    path_brand = []  # path中是否有品牌名称

    # url的domain中相关信息
    domain_count = []  # domain中token数量
    domain_total_length = []  # domain的总长度
    domain_avg_length = []  # domain的平均长度
    domain_max_length = []  # domain的最大长度

    malicious = []  # 是否为恶意域名

    for url in urls:
        url_length.append(url[0])
        # path信息
        path_count.append(list(eval(url[1]))[0])
        path_total_length.append(list(eval(url[1]))[1])
        path_avg_length.append(list(eval(url[1]))[2])
        path_max_length.append(list(eval(url[1]))[3])
        path_brand.append(url[2])
        # domain信息
        domain_count.append(list(eval(url[3]))[0])
        domain_total_length.append(list(eval(url[3]))[1])
        domain_avg_length.append(list(eval(url[3]))[2])
        domain_max_length.append(list(eval(url[3]))[3])
        # domain的字符信息

        # 结果信息
        malicious.append(int(url[4]))

    data_set = {
        'url_length': np.array(url_length),
        'path_count': np.array(path_count),
        'path_avg_length': np.array(path_avg_length),
        'path_total_length': np.array(path_total_length),
        'path_max_length': np.array(path_max_length),
        'path_brand': np.array(path_brand),
        'domain_count': np.array(domain_count),
        'domain_total_length': np.array(domain_total_length),
        'domain_avg_length': np.array(domain_avg_length),
        'domain_max_length': np.array(domain_max_length),
        'malicious': np.array(malicious)

    }

    source_df = DataFrame(data_set) # 转换为DataFrame格式文件

    return source_df

def extract_key_feature_data(source_df):
    """
    从数据集中提取出所有关键特征
    :param source_df: DataFrame 原始数据
    :return:
    """
    sub_columns = source_df.columns.difference(['malicious']) # 求得所有关键特征
    df = source_df[sub_columns]
    y = source_df['malicious']
    return df,y,sub_columns

def train_test_split_data(df,y):
    """
    训练集和测试集分离
    :param df: 原始数据集
    :param y: 原始结果数据集
    :return: 训练数据集，测试数据集，训练结果数据集，测试结果数据集
    """
    from sklearn.cross_validation import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.25,random_state=33)
    return x_train,x_test,y_train,y_test


def cal_decision_tree(x_train,y_train,x_test,y_test):
    """
    决策树计算其成功率
    :param x_train: 训练特征值
    :param y_train: 训练结果值
    :param x_test: 测试特征值
    :param y_test: 测试结果值
    :return:
    """
    from sklearn import tree
    # clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,min_samples_leaf=5)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train,y_train)
    return "{:.2f}".format(clf.score(x_test,y_test)),clf.feature_importances_




def main():
    source_df = create_data_set()
    df,y,sub_columns = extract_key_feature_data(source_df)
    x_train,x_test,y_train,y_test = train_test_split_data(df,y)
    accuracy,feature_importance = cal_decision_tree(x_train,y_train,x_test,y_test)
    print "正确率：",accuracy
    print "各个特征重要性：\n",Series(feature_importance,index= sub_columns)


if __name__ == '__main__':
    main()

# from sklearn import metrics
#
#
# def measure_performance(X,y,clf, show_accuracy=True,
#                         show_classification_report=True,
#                         show_confusion_matrix=True):
#     y_pred=clf.predict(X)
#     if show_accuracy:
#         print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n"
#
#     if show_classification_report:
#         print "Classification report"
#         print metrics.classification_report(y,y_pred),"\n"
#
#     if show_confusion_matrix:
#         print "Confusion matrix"
#         print metrics.confusion_matrix(y,y_pred),"\n"
#
# measure_performance(X_test,y_test,clf, show_classification_report=True, show_confusion_matrix=True)