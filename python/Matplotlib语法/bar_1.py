import numpy as np
from matplotlib import pyplot as plt

# 用bar绘制柱形图
ages_x = [1, 2, 3, 4, 5, 6]

# 设置宽度，默认宽度是0.8
width = 0.1

# 用numpy抓取一个x的范围
x_indexes = np.arange(len(ages_x))
print(x_indexes)

x = [1, 2, 3, 4, 5, 6]

y_1 = [126.02, 114.68, 146.61, 168.56, 65.12, 70.94]
plt.bar(x_indexes - 2 * width, y_1, color='red', width=width, label='R=3')

y_2 = [76.7, 95.71, 101.17, 121.32, 33.74, 60.22]
plt.bar(x_indexes - width, y_2, color='skyblue', width=width, label='R=5')

y_3 = [70.3, 63.1, 80.1, 101.62, 47.74, 65.22]
plt.bar(x_indexes, y_3, width=width, color='darkorange', label='R=7')

y_4 = [80.7, 73.71, 117.17, 126.62, 40.74, 80.22]
plt.bar(x_indexes + width, y_4, color='darkgrey', width=width, label='R=9')


# 当多组数据的柱形图需要并排显示时
plt.xticks(ticks=x_indexes, labels=ages_x)

# x,y轴及图标题
plt.xlabel('Image Sequnce')
plt.ylabel('BSF')

# 唤醒标签
plt.legend()

plt.show()