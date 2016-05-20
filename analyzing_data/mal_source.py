# encoding:utf-8
"""
统计恶意域名来源数量分布
统计各类而已域名的数量分布
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'

# 如果要保存为pdf格式，需要增加如下配置
#rcParams["pdf.fonttype"] = 42

## 统计各个恶意网址公布网站获取域名的数量
# x = [0, 1, 2, 3]
# y = [600,17870,47702,711]
# fig = plt.figure()
# fig.add_subplot(111)
# plt.bar(x,y,align='center')
# # plt.bar(x,y,label=u'恶意域名数量', align='center')
# # plt.legend(prop={'size':11})
# plt.xticks(x,['fedotracker','malwaredomains','phishtank','ransometracker'],fontsize=14)  # x坐标显示内容
# plt.xlabel(u"恶意域名来源")
# plt.ylabel(u"恶意域名数量(个)")

## 统计各个类型的恶意网址
# x = [0, 1, 2]
# y = [47702,17870,1311]
# fig = plt.figure()
# fig.add_subplot(111)
# plt.pie(y,explode=(0,0.05,0),labels=['Phishing','Malware','Other'],autopct='%1.1f%%',colors=['yellowgreen','lightskyblue','gold'],startangle=90)
# plt.axis('equal')

# x = [0, 1, 2, 3, 4, 5]
# y = [1000, 400, 600, 2900, 3000, 2700]
# labels = [u'域名爬虫',u'域名字典',u'关键字爬虫',u'域名生成',u'DNS日志',u'Zone文件']
# colors=['yellowgreen','lightskyblue','gold']
# fig = plt.figure()
# # fig.add_subplot(121)
# # plt.bar(x,y, align='center')
# # plt.xticks(x,labels)  # x坐标显示内容
# # plt.xlabel(u'域名获取方法')
# # plt.ylabel(u"域名数量(万个)")
#
# fig.add_subplot(111)
# plt.pie(y,explode=(0,0.1,0,0,0,0),labels=labels,autopct='%1.1f%%',startangle=90)
# plt.axis('equal')

## 顶级域名数量分布
# x = [0, 1, 2, 3, 4, 5]
# y = [61337635,8673114,8566122,5695915,2355315,20299935]
# labels = ['COM','DE','NET','ORG','RU','OTHER']
# fig = plt.figure()
# fig.add_subplot(121)
# plt.bar(x,y, align='center')
# plt.xticks(x,labels)  # x坐标显示内容
# plt.xlabel(u'顶级域名类型')
# plt.ylabel(u"域名数量（个）")
#
# fig.add_subplot(122)
# plt.pie(y,labels=labels,autopct='%1.1f%%',startangle=90)
# plt.axis('equal')

## 恶意域名顶级域名分布
x = np.arange(15)
y = [15677,1680,1231,691,659,439,436,427,405,388,378,342,260,252,220]
labels = ['com','net','org','uk','ru','au','in','pl','info','ro','it','fr','ca','de','za']
fig = plt.figure()
fig.add_subplot(111)
plt.bar(x,y, align='center')
x_min,x_max = x.min(), x.max()
plt.xlim(x_min-1,x_max+1)
plt.xticks(x,labels)  # x坐标显示内容
plt.xlabel(u'顶级域名类型')
plt.ylabel(u"恶意域名数量（个）")
plt.show()
## 二级whois服务器
# x = np.arange(19)
# y = [3944,1611,1267,682,444,332,280,260,226,223,205,198,197,180,173,150,129,107,102,]
# labels = [
#     'godaddy',
# 'PublicDomainRegistry',
# 'enom',
# 'tucows',
# 'networksolutions',
# 'wildwestdomains',
# 'melbourneit',
# 'onlinenic',
# 'fastdomain',
# 'launchpad',
# 'ovh',
# 'name',
# 'register',
# '1and1',
# 'isimtescil',
# 'hichina',
# 'domain',
# 'rrpproxy',
# 'paycenter'
#
#
# ]

x = np.arange(10)
y = [3944,1611,1267,682,444,332,280,260,226,223]
labels = [
    'godaddy',
'PubDomainReg',
'enom',
'tucows',
'networksolutions',
'wildwestdomains',
'melbourneit',
'onlinenic',
'fastdomain',
'launchpad'
]

fig = plt.figure()
# fig.add_subplot(121)
# plt.bar(x,y, align='center')
# x_min,x_max = x.min(), x.max()
# plt.xlim(x_min-1,x_max+1)
# # plt.xticks(x,labels,rotation=90)  # x坐标显示内容
# plt.xticks(x,labels)  # x坐标显示内容
# plt.xlabel(u'顶级域名类型')
# plt.ylabel(u"恶意域名数量（个）")

fig.add_subplot(111)
plt.pie(y,labels=labels,autopct='%1.1f%%',startangle=90)
plt.axis('equal')

plt.show()

