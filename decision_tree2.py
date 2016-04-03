# encoding:utf-8

from math import log
import operator
from numpy import mean


def get_labels(train_file):
    """
    返回所有数据集类labels(列表)
    """
    labels = []
    for index, line in enumerate(open(train_file, 'rU').readlines()):
        label = line.strip().split(',')[-1]
        labels.append(label)
    return labels


def format_data(dataset_file):
    """
    返回dataset(列表集合)和features(列表)
    """
    dataset = []
    for index, line in enumerate(open(dataset_file, 'rU').readlines()):
        line = line.strip()
        fea_and_label = line.split(',')
        dataset.append(
            [float(fea_and_label[i]) for i in range(len(fea_and_label) - 1)] + [fea_and_label[len(fea_and_label) - 1]])
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    return dataset, features


def split_dataset(dataset, feature_index, best_point):
    """
    按指定feature划分数据集，返回四个列表,划分标准是该特征下所有取值的平均值:
    @dataset_less:指定特征项的属性值＜=该特征项平均值的子数据集
    @dataset_greater:指定特征项的属性值＞该特征项平均值的子数据集
    @label_less:按指定特征项的属性值＜=该特征项平均值切割后子标签集
    @label_greater:按指定特征项的属性值＞该特征项平均值切割后子标签集
    """
    dataset_less = []
    dataset_greater = []
    label_less = []
    label_greater = []
    # datasets = []
    # for data in dataset:
    #     datasets.append(data[0:4])
    # mean_value = mean(datasets, axis=0)[feature_index]  # 数据集在该特征项的所有取值的平均值
    # for data in dataset:
    #     if data[feature_index] > mean_value:
    #
    #         dataset_greater.append(data)
    #         label_greater.append(data[-1])
    #     else:
    #         dataset_less.append(data)
    #         label_less.append(data[-1])
    # return dataset_less, dataset_greater, label_less, label_greater
    # s,point = best_split_point(dataset,feature_index)
    for data in dataset:
        if data[feature_index] > best_point:
            token = data[:feature_index]
            token.extend(data[feature_index+1:])
            dataset_greater.append(token)
            # dataset_greater.append(data)
            label_greater.append(data[-1])
        else:
            token = data[:feature_index]
            token.extend(data[feature_index+1:])
            dataset_less.append(token)
            # dataset_less.append(data)
            label_less.append(data[-1])
    return dataset_less, dataset_greater, label_less, label_greater

def cal_entropy(dataset):
    """
    计算数据集的熵大小
    """
    n = len(dataset)
    if n == 0:
        return 0
    label_count = {}
    for data in dataset:
        label = data[-1]
        if label_count.has_key(label):
            label_count[label] += 1
        else:
            label_count[label] = 1
    entropy = 0
    for label in label_count:
        prob = float(label_count[label]) / n
        entropy -= prob * log(prob, 2)
    return entropy


def cal_info_gain(dataset, feature_index, base_entropy):
    """
    计算指定特征对数据集的信息增益值
    g(D,F) = H(D)-H(D/F) = entropy(dataset) - sum{1,k}(len(sub_dataset)/len(dataset))*entropy(sub_dataset)
    @base_entropy = H(D)
    """
    condition_entropy, point = best_split_point(dataset,feature_index)
    return base_entropy - condition_entropy, point


def best_split_point(dataset,feature_index):
    """
    最好分裂点和增益值
    """
    best_split_info = 10.0
    best_point = -1
    unique_vals = []

    for data in dataset:
        data = data[:-1]
        if data[feature_index] in unique_vals:
            pass
        else:
            unique_vals.append(data[feature_index])
    split_points = find_split_points(unique_vals)
    for split_point in split_points:
        less_sets,more_sets = split_point_data_set(dataset,feature_index, split_point)
        less_prob = len(less_sets)/float(len(dataset))
        more_prob = len(more_sets)/float(len(dataset))
        new_entropy = less_prob * cal_entropy(less_sets) + more_prob*cal_entropy(more_sets)
        if new_entropy <= best_split_info:
            best_split_info = new_entropy
            best_point = split_point
    return best_split_info, best_point


def find_split_points(values):
    """
    寻找分裂点,(i,i+1)/2来计算
    :param values:值
    :return: 分裂点值列表
    """
    split_points = []
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values
    for i in range(0, len(sorted_values)-1):
        split_points.append((float(sorted_values[i])+float(sorted_values[i+1]))/2)
    return split_points


def split_point_data_set(data_set, axis, value):
    """
    分割分裂点的数据集合
    """
    ret_data_set_less = []
    ret_data_set_more = []
    for feat_vec in data_set:
        if feat_vec[axis] <= value:
            ret_data_set_less.append(feat_vec)
        else:
            ret_data_set_more.append(feat_vec)
    return ret_data_set_less, ret_data_set_more


def cal_info_gain_ratio(dataset, feature_index):
    """
    计算信息增益比  gr(D,F) = g(D,F)/H(D)
    """
    base_entropy = cal_entropy(dataset)
    info_gain, point= cal_info_gain(dataset, feature_index, base_entropy)
    info_gain_ratio = info_gain / base_entropy
    return info_gain_ratio, point


def choose_best_fea_to_split(dataset, features):
    '''
    根据每个特征的信息增益比大小，返回最佳划分数据集的特征索引
    '''
    split_fea_index = -1
    max_info_gain_ratio = 0.0
    best_point = -1
    for i in range(len(features)):
        info_gain_ratio,point = cal_info_gain_ratio(dataset, i)
        if info_gain_ratio >= max_info_gain_ratio:
            max_info_gain_ratio = info_gain_ratio
            split_fea_index = i
            best_point = point
    print dataset
    print '特征',features[split_fea_index]
    print "分割点：" + str(best_point)
    print "增益比率" + str(max_info_gain_ratio)
    return split_fea_index,best_point


def most_occur_label(labels):
    """
    返回数据集中出现次数最多的label
    """
    label_count = {}

    for label in labels:
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def build_tree(dataset, labels, features):
    """
    创建决策树
    @dataset:训练数据集
    @labels:数据集中包含的所有label(可重复)
    @features:可进行划分的特征集
    """
    # 若数据集为空,返回NULL
    if len(labels) == 0:
        return 'NULL'
    # 若数据集中只有一种label,返回该label
    if len(labels) == len(labels[0]):
        return labels[0]
    # 若没有可划分的特征集,则返回数据集中出现次数最多的label
    if len(features) == 0:
        return most_occur_label(labels)
    # 若数据集趋于稳定，则返回数据集中出现次数最多的label
    if cal_entropy(dataset) == 0:
        return most_occur_label(labels)
    split_feature_index,best_point = choose_best_fea_to_split(dataset, features)
    split_feature = features[split_feature_index]
    decesion_tree = {split_feature: {}}
    # 若划分特征的信息增益比小于阈值,则返回数据集中出现次数最多的label
    info_gain_ratio,_ = cal_info_gain_ratio(dataset, split_feature_index)
    if info_gain_ratio < 0.2:  # 0.2为阀值
        return most_occur_label(labels)
    del (features[split_feature_index])
    dataset_less, dataset_greater, labels_less, labels_greater = split_dataset(dataset, split_feature_index, best_point)
    decesion_tree[split_feature][str(best_point)+'<=']= build_tree(dataset_less, labels_less, features)
    decesion_tree[split_feature][str(best_point)+'>'] = build_tree(dataset_greater, labels_greater, features)
    return decesion_tree


def store_tree(decesion_tree, filename):
    """
    把决策树以二进制格式写入文件
    """
    import pickle
    writer = open(filename, 'w')
    pickle.dump(decesion_tree, writer)
    writer.close()


def read_tree(filename):
    """
    从文件中读取决策树，返回决策树
    """
    import pickle
    reader = open(filename, 'rU')
    return pickle.load(reader)


def classify(decesion_tree, features, test_data):
    """
    对测试数据进行分类, decesion_tree : {'petal_length': {'<=': {'petal_width': {'<=': 'Iris-setosa', '>': {'sepal_width': {'<=': 'Iris-versicolor', '>': {'sepal_length': {'<=': 'Iris-setosa', '>': 'Iris-versicolor'}}}}}}, '>': 'Iris-virginica'}}
    """

    first_feature = decesion_tree.keys()[0]
    feature_index = features.index(first_feature)
    flag = decesion_tree[first_feature].keys()[0]
    if '<' in flag:
        num = flag.split('<')[0]
    else:
        num = flag.split('>')[0]
    if test_data[feature_index] <= float(num):
        sub_tree = decesion_tree[first_feature][num+"<="]
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data)
        else:
            return sub_tree
    else:
        sub_tree = decesion_tree[first_feature][num+">"]
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data)
        else:
            return sub_tree



def get_means(train_dataset):
    """
    获取训练数据集各个属性的数据平均值
    """
    dataset = []
    for data in train_dataset:
        dataset.append(data[:-1])
    mean_values = mean(dataset, axis=0)  # 数据集在该特征项的所有取值的平均值
    return mean_values


def input():

    data_set = []
    file_name = open('irisTest.txt')
    lines = file_name.readlines()
    for i in lines:
        token = []
        urls = i.strip().split(',')
        token.append(float(urls[0]))
        token.append(float(urls[1]))
        token.append(float(urls[2]))
        token.append(float(urls[3]))
        token.append(urls[4])
        # print token
        data_set.append(token)

    return data_set


def run(train_file, test_file):
    """
    主函数
    """
    labels = get_labels(train_file)
    train_dataset, train_features = format_data(train_file)
    decesion_tree= build_tree(train_dataset, labels, train_features)  # 生成树

    print 'decesion_tree :', decesion_tree
    store_tree(decesion_tree, 'decesion_tree')  # 存储树
    test_dataset, test_features = format_data(test_file)
    n = len(test_dataset)
    correct = 0
    for test_data in test_dataset:
        label = classify(decesion_tree, test_features, test_data)
        if label == test_data[-1]:
            correct += 1
    print "准确率: ", correct / float(n)



if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print "please use: python decision.py train_file test_file"
    #     sys.exit()
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    train_file = 'testa.txt'
    test_file = 'testb.txt'
    run(train_file, test_file)
