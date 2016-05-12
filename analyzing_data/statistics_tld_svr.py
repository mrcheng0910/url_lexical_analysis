# encoding:utf-8
"""
统计顶级域名是否含有域名whois服务器数量
"""

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'  # 支持中文字体
labels = [u'无WHOIS服务器',u'含有WHOIS服务器']
sizes = [321,980]

# colors = ['lightskyblue','yellowgreen']
colors = ['y','c']
explode = (0,0.1)

plt.pie(sizes,labels=labels,colors=colors,explode=explode,shadow=True,startangle=90,autopct='%1.1f%%')
plt.axis('equal')
# plt.plot(x,y)

plt.show()






