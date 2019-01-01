##返回函数
def lazy_sum(*args):
	def sum():
		ax = 0;
		for n in args:
			ax = ax + n;
		return ax;
		pass
	return sum;
	pass

#返回的函数并没有立刻执行，而是直到调用了f()才执行
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()
print(f1(),f2(),f3())