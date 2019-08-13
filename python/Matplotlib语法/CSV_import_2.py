"""
    从CSV文件读取数据作图
"""

import csv
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd


# plt.style.use("fivethirtyeight")

# 使用pandas读取
data = pd.read_csv('/Users/nihaopeng/个人/编程资料/8 Python/code_snippets/Python/Matplotlib/02-BarCharts/data.csv')
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

language_counter = Counter()

for row in lang_responses:
    language_counter.update(row.split(';'))

print(language_counter)
# 输出其中的前10名
print(language_counter.most_common(10))

language = []
popularity = []

for i in language_counter.most_common((10)):
    language.append(i[0])
    popularity.append((i[1]))

# 画横向柱形图
plt.barh(language, popularity)

plt.title('Language')
plt.xlabel('N')
plt.ylabel('L')

plt.show()