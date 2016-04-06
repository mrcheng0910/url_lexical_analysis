# encoding:utf-8

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'

# 如果要保存为pdf格式，需要增加如下配置
#rcParams["pdf.fonttype"] = 42


X = [2000,3000,4000]
DT = [0.92,0.93,0.94]
SVM = [0.83,0.84,0.87]
Logistic = [0.82,0.83,0.94]
KNeighbors = [0.72,0.75,0.76]
GaussianNB = [0.6,0.65,0.65]

plt.plot(X,DT,label='y2',color='r',linewidth=2)
plt.plot(X,SVM,label='y3',color='b',linewidth=2)
plt.plot(X,Logistic,label='y4',color='k',linewidth=2)
plt.plot(X,KNeighbors,label='y5',color='c',linewidth=2)
plt.plot(X,GaussianNB,label='y5',color='m',linewidth=2)


plt.xlabel(u'数据集网址数量(个)')
plt.ylabel(u'预测正确率')
plt.title(u'五种分类算法正确率比较')
plt.legend()

plt.show()

# plt.bar(x,y,label="one",color='blue') # 住装图
# plt.bar(x2,y2,label="sec",color='red')

# plt.plot(x2,y2,label="second") # 线装图
# plt.plot(x,y,label="first")

# population_ages = [22,23,35,45,32,65,44,12,110,130,33,45,67,87,33,33]

# ids = [x for x in range(len(population_ages))]

# bins = [0,10,20,30,40,50,60,70,80,90,100,120,130,140]
#
# plt.hist(population_ages,bins,histtype='bar',rwidth=0.8,label='populat') # 直方图

# x = [1,2,3,4,5,6,7,8]
#
# y = [2,5,6,7,5,4,7,5]
# plt.scatter(x,y,label="skitscat",color='k',marker="*",s=50) # 散点图

# days = [1,2,3,4,5]
# sleeping = [5,6,4,8,2]
# eating = [2,3,5,6,5]
# working = [7,8,9,6,6]
# playing = [5,6,7,8,7]
# plt.plot([],[],color='m',label='sleeping',linewidth=3)
# plt.plot([],[],color='c',label='eating')
# plt.plot([],[],color='r',label='woring')
# plt.plot([],[],color='k',label='playing')
#
# plt.stackplot(days,sleeping,eating,working,playing,colors=['m','c','r','k']) #堆叠式图区

# slices = [7,2,2,13]
# activities = ['sleeping','eating','working','playing']
# colors = ['c','m','r','b']
# plt.pie(slices,
#         labels=activities,
#         colors=colors,
#         startangle=90,
#         shadow=True,
#         explode=(0,0.1,0,0),
#         autopct='%1.1f%%'
#         )



