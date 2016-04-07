# encoding:utf-8
"""
混淆矩阵展示
"""

from datamining import test
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'  # 支持中文字体
# 如果要保存为pdf格式，需要增加如下配置
#rcParams["pdf.fonttype"] = 42


def plot_confusion_matrix(cm, title=u'SVM分类恶意域名混淆矩阵图', cmap=plt.cm.Blues):
    """
    绘制矩阵图
    :param cm:
    :param title:
    :param cmap:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [u'恶意网址',u'良性网址'])
    plt.yticks(tick_marks, [u'恶意网址',u'良性网址'], rotation=90)
    plt.tight_layout()
    plt.ylabel(u'真实结果')
    plt.xlabel(u'预测结果')
    # 设置数据标签
    tp = cm[0][0]  # good-->good
    fp = cm[0][1]  # good-->bad
    fn = cm[1][0]  # bad-->good
    tn = cm[1][1]  # bad-->bad
    font_size = 15  # 文本字体大小
    tp_label = plt.text(0.30, 0.40, tp)
    plt.setp(tp_label, color='w', fontsize=font_size)
    fp_label = plt.text(0.57, 0.40, fp)
    plt.setp(fp_label, color='k',fontsize=font_size)
    fn_label = plt.text(0.30, 0.65, fn)
    plt.setp(fn_label, color='k',fontsize=font_size)
    tn_label = plt.text(0.57, 0.65, tn)
    plt.setp(tn_label, color='w',fontsize=font_size)


# Compute confusion matrix
# cm = confusion_matrix(y_test, y_pred)
cm = test()

np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
title = u''
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)
# plt.figure()
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()