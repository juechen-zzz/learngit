"""
    www.youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_
    第一讲：matplotlib 基础
        color：颜色
        linestyle：线型
        marker：作图点标记
        linewidth：线宽
        label：标签
"""

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

print(plt.style.available)

ages_x = [20, 21, 22, 23, 24, 25, 26, 27]

# label作用是图片折线对应的含义标签，设置黑色虚线
dev_y = [12412, 45235, 12411, 45623, 47231, 46732, 12677, 34561]
# plt.plot(ages_x, dev_y, 'k--', label='All star')
plt.plot(ages_x, dev_y, color='k', linestyle='--', marker='*', linewidth=3, label='All star')

# 可以共享同一个横轴，设置颜色(改为十六进制值)、线条，蓝色实线
py_dev_y = [51434, 23455, 45677, 32451, 45748, 12478, 75689, 12344]
plt.plot(ages_x, py_dev_y, color='#5a7d9a', marker='o', label='Python')

# x,y轴及图标题
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Figure 1')

# 唤醒标签
plt.legend()

# 添加网格
plt.grid(True)

# 一种填充方法
plt.tight_layout()

# 显示
plt.show()

