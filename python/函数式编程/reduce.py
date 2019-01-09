#把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算
#python 3.0以后就必须引用reduce，不在默认函数中
from functools import reduce
def fn(x,y):
	return x*10 +y;
	pass
s = reduce(fn,[1,2,3,4]);
print(s);

def char2num(m):
	d = {'1':1,'2':2};
	return d[m]
	pass
a = reduce(fn,map(char2num,'12'));
print(a);