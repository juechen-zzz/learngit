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

y_1 = [70.37, 63.71, 90.17, 71.62, 107.74, 85.22]
plt.bar(x_indexes - 2 * width, y_1, color='red', width=width, label='R=3')

y_2 = [46.7, 55.71, 81.17, 61.32, 113.74, 60.22]
plt.bar(x_indexes - width, y_2, color='skyblue', width=width, label='R=5')

y_3 = [40.3, 53.1, 40.1, 41.62, 87.74, 35.22]
plt.bar(x_indexes, y_3, width=width, color='darkorange', label='R=7')

y_4 = [30.7, 23.71, 17.17, 26.62, 70.74, 30.22]
plt.bar(x_indexes + width, y_4, color='darkgrey', width=width, label='R=9')


# 当多组数据的柱形图需要并排显示时
plt.xticks(ticks=x_indexes, labels=ages_x)

# x,y轴及图标题
plt.xlabel('Image Sequnce')
plt.ylabel('STCR')

# 唤醒标签
plt.legend()

plt.show()