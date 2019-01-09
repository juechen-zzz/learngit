#需要用循环变量时的返回函数
def count():
	def f(j):
		def g():
			return j*j
		return g;
	fs = [];
	for i in range(1,4):
		fs.append(f(i));
	return fs;

f1, f2, f3 = count()
print(f1(),f2(),f3())`