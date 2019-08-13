"""
    第二讲：柱形图基础
        plot:折线图
        bar:柱形图
        barh：横向柱形图（注意改横轴纵轴的名字）
"""
import numpy as np
from matplotlib import pyplot as plt

# 用bar绘制柱形图
ages_x = [20, 21, 22, 23, 24, 25, 26, 27]
# 设置宽度，默认宽度是0.8
width = 0.25

# 用numpy抓取一个x的范围
x_indexes = np.arange(len(ages_x))
print(x_indexes)


dev_y = [12412, 45235, 12411, 45623, 47231, 46732, 12677, 34561]
plt.bar(x_indexes - width, dev_y, width=width, color='#444444', label='All Star')

py_dev_y = [51434, 23455, 45677, 32451, 45748, 12478, 75689, 12344]
plt.bar(x_indexes, py_dev_y, width=width, color='#008fd5', label='Python')

# 当多组数据的柱形图需要并排显示时
plt.xticks(ticks=x_indexes, labels=ages_x)

# x,y轴及图标题
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Figure 1')

# 唤醒标签
plt.legend()

plt.show()