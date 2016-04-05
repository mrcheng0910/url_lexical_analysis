#!/usr/bin/python
# encoding:utf-8

"""
分析通过决策树的效果特征
"""


import pandas as pd
df = pd.read_csv('train.csv')

# print df.head()

subdf = df[['Pclass','Sex','Age']]
y = df.Survived

age = subdf['Age'].fillna(value=subdf.Age.mean())
pclass = pd.get_dummies(subdf['Pclass'],prefix='Pclass')
sex = (subdf['Sex']=='male').astype('int')
X = pd.concat([pclass,age,sex],axis=1)
# print X.head()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print X_train
print y_train
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)
# clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,y_train)
print "准确率为：{:.2f}".format(clf.score(X_test,y_test))

print clf.feature_importances_

# #############################
# from sklearn import metrics

# def measure_performance(X,y,clf, show_accuracy=True,
#                         show_classification_report=True,
#                         show_confusion_matrix=True):
#     y_pred=clf.predict(X)
#     if show_accuracy:
#         print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n"

#     if show_classification_report:
#         print "Classification report"
#         print metrics.classification_report(y,y_pred),"\n"

#     if show_confusion_matrix:
#         print "Confusion matrix"
#         print metrics.confusion_matrix(y,y_pred),"\n"

# measure_performance(X_test,y_test,clf, show_classification_report=True, show_confusion_matrix=True)