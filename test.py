#!/usr/bin/python
# encoding:utf-8

"""
使用ID3算法，根据恶意域名的关键词特征，来计算最优树
"""

from math import log
import operator
from data_base import MySQL


def fetch_data():
    """
    从数据库中获取数据基础数据
    :return: 返回基础数据
    """
    db = MySQL()
    sql = 'SELECT url_length,domain_tokens,malicious FROM url_features limit 1500'
    db.query(sql)
    urls = db.fetchAllRows()
    db.close()
    return urls


def create_data_set(urls):
    """
    根据获取的基础数据，建立对应的测试数据集合
    :param urls: 基础数据
    :return: 数据集合
    """
    data_set = []
    for url in urls:
        domain_token = []
        domain_token.append(url[0])
        domain_token.extend(list(eval(url[1])))
        domain_token.append(url[2])
        data_set.append(domain_token)

    labels = ['url_length','token_count','total_length','avg_length','max_length']  # 特征标签


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


def find_split_points(values):
    """
    寻找分裂点,(i,i+1)/2来计算
    :param values:值
    :return: 分裂点值列表
    """
    split_points = []
    sorted_values = sorted(values)
    for i in range(0,len(sorted_values)-1):
        split_points.append((int(sorted_values[i])+int(sorted_values[i+1]))/2)

    return split_points


def cal_best_split_point(data_set, unique_vals, num_feature):
    """
    计算最优分类点
    :param data_set: 测试数据集
    :param unique_vals: 子集的值
    :param num_feature: 特征编号
    :return: 返回最优分裂点值和分裂点编号
    """
    best_split_info = 10
    best_split_point = 0
    split_points = find_split_points(unique_vals)
    for split_point in split_points:
        less_sets,more_sets = split_point_data_set(data_set, num_feature, split_point)
        less_prob = len(less_sets)/float(len(data_set))
        more_prob = len(more_sets)/float(len(data_set))
        new_entropy = less_prob * calc_shannon_ent(less_sets)+more_prob*calc_shannon_ent(more_sets)
        if new_entropy < best_split_info:
            best_split_info = new_entropy
            best_split_point = split_point
    return best_split_info, best_split_point


def split_point_data_set(data_set, axis, value):
    """
    分割分裂点的数据集合
    :param data_set:
    :param axis: 特征编号编号
    :param value: 分裂点值
    :return: 返回小于等于分裂点值的集合和大于分裂点值的集合
    """
    ret_data_set_less = []
    ret_data_set_more = []
    for feat_vec in data_set:
        if feat_vec[axis] <= value:
            reduced_feat_vec = feat_vec[:axis]  # chop out axis used for splitting
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set_less.append(reduced_feat_vec)
        else:
            reduced_feat_vec = feat_vec[:axis]  # chop out axis used for splitting
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set_more.append(reduced_feat_vec)
    return ret_data_set_less, ret_data_set_more


def split_data_set(data_set, axis, value):
    """
    :param data_set: 数据集合
    :param axis:  特征编号
    :param value: 数据值
    :return: 返回分开后的数据集合
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]          # chop out axis used for splitting
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def get_info_gain(data_set,unique_vals,num_feature):
    """
    计算信息增益
    :param data_set: 数据集
    :param unique_vals: 特征值
    :param num_feature: 特征编号
    :return: 增益和分裂点
    """
    split_info,split_point = cal_best_split_point(data_set,unique_vals,num_feature)
    base_info = calc_shannon_ent(data_set)
    return base_info - split_info,split_point


def choose_best_feature_to_split(data_set):
    """
    根据当前数据集中的特征，来进行最好的划分方式
    :param data_set:需要划分的数据集
    :return: 返回特征编号
    """

    num_features = len(data_set[0]) - 1      # 获取数据集特征数量

    best_info_gain = 0.0  # 初始化最优信息增益
    best_feature = -1  # 初始化最优特征编号
    split_point = 0
    for i in range(num_features):  # 循环查询所有特征
        feat_list = [example[i] for example in data_set]  # 创建某特征的样本值列表
        # print feat_list
        unique_vals = set(feat_list)       # 得到特征的所有不重复值
        info_gain,split_point = get_info_gain(data_set,unique_vals,i)
        if info_gain > best_info_gain:  # 比较最优信息增益
            best_info_gain = info_gain  # 如果大，则为最优信息增益
            best_feature = i  # 最优特征

    return best_feature, split_point  # 返回最优特征编码


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
    创建决策树
    :param data_set: 数据集合
    :param labels: 特征标签
    :return:
    """
    class_list = [example[-1] for example in data_set]  # 获得所有类别值
    if class_list.count(class_list[0]) == len(class_list):  # 如果所有类别值相同，则停止递归，返回结果
        return class_list[0]

    if len(data_set[0]) == 1:  # 遍历完所有特征后，只剩下类别值，则采用多数表决的方式。
        return majority_cnt(class_list)

    best_feat,split_point = choose_best_feature_to_split(data_set)  # 最优特征号
    best_feat_label = labels[best_feat]  # 最优特征标签
    my_tree = {best_feat_label: {}}
    del labels[best_feat]  # 删除最优特征

    new_data_set = []
    for i in data_set:
        if i[best_feat]<=split_point:
            i[best_feat]='<='+str(split_point)
            new_data_set.append(i)
        else:
            i[best_feat]='>'+str(split_point)
            new_data_set.append(i)

    feat_values = [example[best_feat] for example in new_data_set]
    unique_vals = set(feat_values)

    for value in unique_vals:
        sub_labels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        my_tree[best_feat_label][value] = create_tree(split_data_set(new_data_set, best_feat, value), sub_labels)

    return my_tree


if __name__ == '__main__':

    urls = fetch_data()
    data_set,labels = create_data_set(urls)
    print create_tree(data_set,labels)

