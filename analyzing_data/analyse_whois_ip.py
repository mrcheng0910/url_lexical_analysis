#encoding:utf-8

"""
分析whois的ip地址的分布
"""


from data_base import MySQL
import numpy as np

DBCONFIG = {'host':'172.26.253.3',
                'port': 3306,
                'user':'root',
                'passwd':'platform',
                'db':'DomainWhois',
                'charset':'utf8'}



db = MySQL(DBCONFIG)


sql = 'select port_available from svr_ip WHERE  level="2"'

db.query(sql)

test = db.fetch_all_rows()

result = []

for i in test:

    result.append(len(i[0]))


# print result

result = np.array(result)

# print np.unique(result)
# print len(result[result==1])
x = []
y = []
for i in np.unique(result):
    print i,len(result[result==i])
    x.append(i)
    y.append(len(result[result==i]))


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

import matplotlib
# matplotlib.rcParams['text.color'] = 'w'
## 一级whois服务器统计
# x = [0, 1, 2, 3]
# y = [273, 69, 11, 29]
# labels = ['1','2','3-10','>10']
# fig = plt.figure()
# fig.add_subplot(121)
# plt.bar(x,y, align='center')
# plt.xticks(x,labels)  # x坐标显示内容
# plt.xlabel(u'IP个数')
# plt.ylabel(u"WHOIS服务器数量")
#
# fig.add_subplot(122)
# plt.pie(y,labels=labels,autopct='%1.1f%%',startangle=90)
# plt.axis('equal')
#
# plt.show()

x = [0, 1, 2]
y = [928, 33, 10]
labels = ['1','2','>3']
fig = plt.figure()
fig.add_subplot(121)
plt.bar(x,y, align='center')
plt.xticks(x,labels)  # x坐标显示内容
plt.xlabel(u'IP个数')
plt.ylabel(u"WHOIS服务器数量")

fig.add_subplot(122)
plt.pie(y,labels=labels,explode=(0, 0, 0.05),autopct='%1.1f%%',startangle=90,colors=['g','b','m'])
plt.axis('equal')

plt.show()



















