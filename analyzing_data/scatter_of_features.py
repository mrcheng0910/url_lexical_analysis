#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

from analyzing_data.datamining import extract_key_feature_data

# x = [1,2,3,4,5]

# y = [2, 5, 6, 7, 5]
# y1 = [1, 3, 5, 6, 7]

df,y,sub_columns = extract_key_feature_data()

length = len(sub_columns)*2
x0 = [i for i in range(0,length,2)]
x1 = [i for i in range(1,length,2)]
# print x
# print df.ix[0].values
# for i in range(len(df)):
#     # print df.ix[i].values
#     if y.ix[i] == 0:
#         plt.scatter(x0,df.ix[i].values,color='c')
#     else:
#         plt.scatter(x1,df.ix[i].values,color='k')
#
# # plt.scatter(x,y,label="skitscat") # 散点图
# # plt.scatter(x,y1,label="skitscat") # 散点图
#
plt.show()