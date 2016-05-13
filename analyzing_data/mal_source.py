# encoding:utf-8
"""
统计恶意域名来源数量分布
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'

# 如果要保存为pdf格式，需要增加如下配置
#rcParams["pdf.fonttype"] = 42

x = [0, 1, 2, 3]
y = [600,17870,47702,711]
fig = plt.figure()
fig.add_subplot(111)
plt.bar(x,y,align='center')
# plt.bar(x,y,label=u'恶意域名数量', align='center')
# plt.legend(prop={'size':11})
plt.xticks(x,['fedo_tracker','malwaredomains','phishtank','ransome_tracker'])  # x坐标显示内容
plt.xlabel(u"恶意域名来源")
plt.ylabel(u"恶意域名数量(个)")

plt.show()

