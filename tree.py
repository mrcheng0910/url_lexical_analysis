# encoding:utf-8
"""
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
"""

from math import log
import operator
from data_base import MySQL


def fetch_data():
    """
    获取数据
    :return:
    """
    db = MySQL()
    sql = 'SELECT domain_tokens,malicious FROM url_features'
    db.query(sql)
    urls = db.fetchAllRows()
    db.close()
    return urls


def create_data_set(urls):
    """
    生成测试数据集
    :return:
    """
    data_set = []
    for url in urls:

        # print list(url[0]).extend(url[1])
        # data_set.append(url[0])
        domain_token = list(eval(url[0]))
        domain_token.append(url[1])
        # print domain_token[0],domain_token[1],domain_token[2],domain_token
        data_set.append(domain_token)




    # data_set = [[1, 1, 'yes'],
    #            [1, 1, 'yes'],
    #            [1, 0, 'no'],
    #            [0, 1, 'no'],
    #            [0, 1, 'no']]

    labels = ['a','b','c','d']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算数据集的熵,并返回熵的值
    :param data_set: 数据集
    :return:shannon_ent 熵值
    """
    num_entries = len(data_set)  # 数据集数量
    label_counts = {}

    for featVec in data_set:  # 得到数据集所有结果类型
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_ent -= prob * log(prob,2)  # 计算熵
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    分割出特征
    :param data_set:
    :param axis:
    :param value:
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]  # chop out axis used for splitting
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    # print ret_data_set
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的特征划分方式
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1      # 获取数据集特征数量
    base_entropy = calc_shannon_ent(data_set)  # 原始熵
    best_info_gain = 0.0  # 最优信息增益
    best_feature = -1  # 最优特征
    for i in range(num_features):  # 循环查询所有特征
        feat_list = [example[i] for example in data_set]  # 创建某特征的样本列表
        print len(feat_list)
        unique_vals = set(feat_list)       # 得到特征的所有不重复值
        if len(unique_vals)>=3:
            print sorted(unique_vals)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy     #计算信息增益

        # print info_gain
        if info_gain > best_info_gain:  # 比较最优信息增益
            best_info_gain = info_gain  # 如果大，则为最优信息增益
            best_feature = i  # 最优特征

    # print best_feature
    return best_feature                      # 返回最优特征编码


def majority_cnt(class_list):
    """
    多数表均决定叶子结点的类型
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 如果有相同的票数呢？要考虑的地方


def create_tree(data_set, labels):
    """

    :param data_set:
    :param labels:
    :return:
    """
    class_list = [example[-1] for example in data_set]  # 获得所有类别值
    # print class_list
    if class_list.count(class_list[0]) == len(class_list):  # 如果所有类别值相同，则停止递归，返回结果
        return class_list[0]

    if len(data_set[0]) == 1:  # 遍历完所有特征后，只剩下类别值，则采用多数表决的方式。
        return majority_cnt(class_list)

    best_feat = choose_best_feature_to_split(data_set)  # 最优特征号
    # best_feat_label = labels[best_feat]  # 最优特征标签
    #
    # my_tree = {best_feat_label:{}}
    #
    # del labels[best_feat]  # 删除最优特征
    #
    # feat_values = [example[best_feat] for example in data_set]
    # unique_vals = set(feat_values)
    #
    # for value in unique_vals:
    #     sub_labels = labels[:]       #copy all of labels, so trees don't mess up existing labels
    #     my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)

    # return my_tree
    return 0


def classify(input_tree, feat_labels, test_vec):
    """
    分类，测试数据
    :param input_tree:
    :param feat_labels:
    :param test_vec:
    :return:
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        classLabel = classify(value_of_feat, feat_labels, test_vec)
    else: classLabel = value_of_feat
    return classLabel


def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(file_name):
    import pickle
    fr = open(file_name)
    return pickle.load(fr)



if __name__ == '__main__':
    # data_set,labels = create_data_set()
    # print create_tree(data_set,labels)
    urls = fetch_data()
    data_set,labels = create_data_set(urls)
    print create_tree(data_set,labels)
    # calc_shannon_ent(data_set)
    # split_data_set(data_set, 0, 1)
    # split_data_set(data_set, 0, 0)
    # choose_best_feature_to_split(dataSet)
