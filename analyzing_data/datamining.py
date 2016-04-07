#!/usr/bin/python
# encoding:utf-8

"""
使用多种分类算法对恶意域名分类效果进行统计分析
使用恶意域名的15个特征
"""
from data_base import MySQL
import numpy as np
from pandas import DataFrame,Series
from sklearn import metrics
from sklearn import cross_validation
from sklearn import tree  # 决策树
from sklearn.svm import SVC  # SVC支持向量机
from sklearn.neighbors import KNeighborsClassifier  # K近邻算法


def fetch_data():
    """
    从数据库中获取数据基础数据
    :return: 返回基础数据
    """
    db = MySQL()
    sql = 'SELECT url_length,path_tokens,path_brand,domain_tokens,malicious,domain_characters,path_characters FROM url_features'
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

    # url 信息
    url_length = []  # url的长度

    # url的path中相关信息
    path_count = []  # path中token数量
    path_total_length = []  # path的总长度
    path_avg_length = []  # path的平均长度
    path_max_length = []  # path的最大长度
    path_brand = []  # path中是否有品牌名称
    path_digit = []  # path中数字的个数
    path_alapha = []  # path中字母的个数
    path_special_character = []  # path中特殊字符的个数

    # url的domain中相关信息
    domain_count = []  # domain中token数量
    domain_total_length = []  # domain的总长度
    domain_avg_length = []  # domain的平均长度
    domain_max_length = []  # domain的最大长度
    domain_digit = []  # domain中含有数字个数
    domain_alapha = []  # domain中字母的个数
    domain_special_character = []  # domain中特殊字符的个数
    malicious = []  # 是否为恶意域名

    for url in urls:
        url_length.append(url[0])
        # path长度信息
        path_count.append(list(eval(url[1]))[0])
        path_total_length.append(list(eval(url[1]))[1])
        path_avg_length.append(list(eval(url[1]))[2])
        path_max_length.append(list(eval(url[1]))[3])
        path_brand.append(url[2])
        # domain长度信息
        domain_count.append(list(eval(url[3]))[0])
        domain_total_length.append(list(eval(url[3]))[1])
        domain_avg_length.append(list(eval(url[3]))[2])
        domain_max_length.append(list(eval(url[3]))[3])
        # domain的字符信息
        domain_digit.append(list(eval(url[5]))[0])
        domain_special_character.append(list(eval(url[5]))[1])
        domain_alapha.append(sum(list(eval(url[5]))[2:]))
        # path的字符信息
        path_digit.append(list(eval(url[6]))[0])
        path_special_character.append(list(eval(url[6]))[1])
        path_alapha.append(sum(list(eval(url[6]))[2:]))
        # 结果信息
        malicious.append(int(url[4]))
        # malicious.append(str(url[4]))

    data_set = {
        'url_length': np.array(url_length).astype(float),
        'path_count': np.array(path_count).astype(float),
        'path_avg_length': np.array(path_avg_length).astype(float),
        'path_total_length': np.array(path_total_length).astype(float),
        'path_max_length': np.array(path_max_length).astype(float),
        'path_brand': np.array(path_brand).astype(float),
        'domain_count': np.array(domain_count).astype(float),
        'domain_total_length': np.array(domain_total_length).astype(float),
        'domain_avg_length': np.array(domain_avg_length).astype(float),
        'domain_max_length': np.array(domain_max_length).astype(float),
        'malicious': np.array(malicious),
        'domain_digit': np.array(domain_digit).astype(float),
        'domain_special_character': np.array(domain_special_character).astype(float),
        # 'domain_alapha': np.array(domain_alapha),
        'path_digit': np.array(path_digit).astype(float),
        'path_special_character': np.array(path_special_character).astype(float),
        # 'path_alapha': np.array(path_alapha),

    }

    source_df = DataFrame(data_set) # 转换为DataFrame格式文件

    return source_df

def extract_key_feature_data():
    """
    从数据集中提取出所有关键特征
    :param source_df: DataFrame 原始数据
    :return:
    """
    source_df = create_data_set()
    sub_columns = source_df.columns.difference(['malicious']) # 求得所有关键特征
    df = source_df[sub_columns]
    y = source_df['malicious']
    return df,y,sub_columns

def train_test_split_data(df,y,train_size=0.80):
    """
    训练集和测试集分离
    :param df: 原始数据集
    :param y: 原始结果数据集
    :return: 训练数据集，测试数据集，训练结果数据集，测试结果数据集
    """
    from sklearn.cross_validation import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(df,y,train_size=train_size,random_state=33)
    return x_train,x_test,y_train,y_test


def cal_decision_tree(x_train,y_train,x_test,y_test):
    """
    决策树计算其成功率
    :param x_train: 训练特征值
    :param y_train: 训练结果值
    :param x_test: 测试特征值
    :param y_test: 测试结果值
    :return:正确率，重要程度
    """

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,min_samples_leaf=7)
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(x_train,y_train)
    return "{:.4f}".format(clf.score(x_test,y_test)), clf.feature_importances_,clf


def logistic_data(X,y):
    """
    逻辑回归算法
    :param X:
    :param y:
    :return:
    """
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)
    score = model.score(X,y)
    cm = metrics.confusion_matrix(expected, predicted)
    # print(metrics.classification_report(expected, predicted,labels=[0,1],target_names=['良性网址','恶意网址']))
    return score, cm


def svm_data(X,y):
    """
    使用支持向量机分析数据
    :param X: 训练数据集
    :param y: 结果数据集
    """

    # fit a SVM model to the data
    model = SVC()
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)
    score = model.score(X,y)
    cm = metrics.confusion_matrix(expected, predicted)
    # print metrics.classification_report(expected, predicted, labels=[0,1],target_names=['良性网址','恶意网址'])
    return score, cm


def k_neighbor_data(X,y):
    """
    k邻接算法
    :param X:
    :param y:
    :return:
    """
    # fit a k-nearest neighbor model to the data
    model = KNeighborsClassifier()
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)
    score = model.score(X,y)
    # print(metrics.classification_report(expected, predicted,labels=[0,1],target_names=['良性网址','恶意网址']))
    cm = metrics.confusion_matrix(expected, predicted)
    return score,cm


def gausssian_data(X,y):
    """
    朴素贝叶斯算法
    :param X:
    :param y:
    :return:
    """
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X, y)
    expected = y
    predicted = model.predict(X)
    score = model.score(X,y)
    # print(metrics.classification_report(expected, predicted,labels=[0,1],target_names=['良性网址','恶意网址']))
    cm = metrics.confusion_matrix(expected, predicted)
    return score, cm



def measure_performance(X,y,clf, show_accuracy=True,
                        show_classification_report=True,
                        show_confusion_matrix=True):
    """
    多指标来评估模型
    :param X: 测试集
    :param y: 真实结果
    :param clf: 模型
    :param show_accuracy: 显示正确率
    :param show_classification_report: 显示分类报告
    :param show_confusion_matrix:
    :return:
    """
    y_pred = clf.predict(X)
    if show_accuracy:
        print "Accuracy:{0:.4f}".format(metrics.accuracy_score(y,y_pred)), "\n"

    if show_classification_report:
        print "模型分类报告："
        print metrics.classification_report(y,y_pred,labels=[0,1],target_names=['良性网址','恶意网址']), "\n"

    if show_confusion_matrix:
        print "混淆矩阵报告："
        print metrics.confusion_matrix(y,y_pred),"\n"


def cross_validation_model(clf,df,y,cv=10):
    """
    交叉验证模型
    :param clf: 模型
    :param df: 数据
    :param y: 结果数据
    :param cv: 交叉验证次数
    :return: 验证分数列表，平均值，标准差
    """
    scores = cross_validation.cross_val_score(clf, df, y, cv=10)
    return  scores,scores.mean(),scores.std()





def draw_tree(clf,sub_columns):
    print "绘制图"
    import pydot,StringIO
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,feature_names=sub_columns,max_depth=5)
    dot_data.getvalue()
    pydot.graph_from_dot_data(dot_data.getvalue())
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('titanic.png')
    # from IPython.core.display import Image
    # Image(filename='titanic.png')




def main():

    df,y,sub_columns = extract_key_feature_data()
    x_train,x_test,y_train,y_test = train_test_split_data(df,y,train_size=0.90)
    print len(x_train)
    accuracy,feature_importance,clf = cal_decision_tree(x_train,y_train,x_test,y_test)
    print "正确率：",accuracy
    print "各个特征重要性：\n",Series(feature_importance,index= sub_columns)
    measure_performance(x_test,y_test,clf, show_classification_report=True, show_confusion_matrix=True)
    # scores,scores_mean,scores_std = cross_validation_model(clf,df,y,cv=10)
    # print "验证结果分数列表", scores
    # print "平均值：", scores_mean
    # print "标准偏差估计分数：", scores_std
    # draw_tree(clf,sub_columns)

    # from sklearn.ensemble import RandomForestClassifier
    # clf2 = RandomForestClassifier(n_estimators=1000,random_state=33)
    # clf2 = clf2.fit(x_train,y_train)
    # scores2 = cross_validation.cross_val_score(clf2,df, y, cv=5)
    # print scores2.mean()
    # print clf2.feature_importances_

    # guiyih(df,y)
    svm_data(df,y)
    k_neighbor_data(df,y)
    gausssian_data(df,y)
    logistic_data(df,y)

if __name__ == '__main__':
    main()

