"""
    从CSV文件读取数据作图
"""

import csv
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


# plt.style.use("fivethirtyeight")

# 读取文件并计数，返回一个Counter字典
with open('/Users/nihaopeng/个人/编程资料/8 Python/code_snippets/Python/Matplotlib/02-BarCharts/data.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    # 对csv中出现的字符串种类进行计数
    language_counter = Counter()

    for row in csv_reader:
        language_counter.update(row['LanguagesWorkedWith'].split(';'))
    """
    row = next(csv_reader)
    print(row)  # OrderedDict([('Responder_id', '1'), ('LanguagesWorkedWith', 'HTML/CSS;Java;JavaScript;Python')])
    print(row['LanguagesWorkedWith'])   # HTML/CSS;Java;JavaScript;Python
    print(row['LanguagesWorkedWith'].split(';'))    # ['HTML/CSS', 'Java', 'JavaScript', 'Python']
    """
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