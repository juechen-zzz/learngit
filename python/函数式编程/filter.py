#filter()函数用于过滤序列
#识别奇偶
def is_odd(n):
	return n % 2 ==0
	pass
print(list(filter(is_odd,[1,2,3,4])));

#删除空字符
def not_empty(s):
	return s and s.strip()
	pass
print(list(filter(not_empty, ['A', '', 'B', None, 'C', '  '])))