#encoding:utf-8

"""
分析恶意域名domain和非恶意域名domain的长度分布
"""


from data_base import MySQL
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
import numpy as np

# #
# db = MySQL()
# mal_sql = 'select domain_tokens from url_features WHERE malicious="1"'
# db.query(mal_sql)
# mal = db.fetch_all_rows()
# mal_y = []
# for i in mal:
#     mal_y.append(list(eval(i[0]))[1])
#
# beg_sql = 'select domain_tokens from url_features WHERE malicious="0"'
# db.query(beg_sql)
# beg = db.fetch_all_rows()
# beg_y = []
# for i in beg:
#     beg_y.append(list(eval(i[0]))[1])
# a = np.array(mal_y)
# b = np.array(beg_y)
#
# print len(b[b<20])
# print len(b[b>=20])
# print a.mean()
# print b.mean()
#
# mal_x = np.arange(len(mal_y))
# beg_x = np.arange(len(beg_y))
# fig = plt.figure()
# fig.add_subplot(121)
# plt.plot(beg_x,beg_y)
# plt.xlabel(u'网址个数')
# plt.ylabel(u"域名长度")
# fig.add_subplot(122)
# plt.plot(mal_x,mal_y,color='r')
# plt.xlabel(u'网址个数')
# plt.ylabel(u"域名长度")
# plt.show()



db = MySQL()
mal_sql = 'select domain_tokens,domain_characters from url_features WHERE malicious="1"'
db.query(mal_sql)
mal = db.fetch_all_rows()
mal_y = []
for j,i in mal:
    # print j,i
    mal_y.append(int(list(eval(i))[1]*list(eval(j))[1]/100.0))

beg_sql = 'select domain_tokens,domain_characters from url_features WHERE malicious="0"'
db.query(beg_sql)
beg = db.fetch_all_rows()
beg_y = []
for j,i in beg:
    beg_y.append(int(list(eval(i))[1]*list(eval(j))[1]/100.0))

# print mal_y

#
#
#
mal_x = np.arange(len(mal_y))
beg_x = np.arange(len(beg_y))
mal_y.sort()
# beg_y.sort()
# print mal_y
fig = plt.figure()
fig.add_subplot(121)
plt.plot(beg_x,beg_y)
plt.xlabel(u'网址个数')
plt.ylabel(u"域名中出现数字次数")
fig.add_subplot(122)
plt.plot(mal_x,mal_y,color='r')
plt.xlabel(u'网址个数')
plt.ylabel(u"域名中出现数字次数")
plt.show()
















