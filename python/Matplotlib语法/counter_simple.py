"""
    counter：计数，计算出现的次数
"""

from collections import Counter

c = ['a', 'b', 'c', 'd', 'e', 'd']
a = Counter(c)
print(a)    # Counter({'d': 2, 'a': 1, 'b': 1, 'c': 1, 'e': 1})

# 再次添加数据
a.update(['c'])
print(a)    # Counter({'c': 2, 'd': 2, 'a': 1, 'b': 1, 'e': 1})
