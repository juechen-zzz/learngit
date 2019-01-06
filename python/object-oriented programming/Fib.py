class Fib(object):
	def __init__(self):
		self.a,self.b = 0,1		# 初始化两个计数器a，b

	def __iter__(self):
		return self				# 实例本身就是迭代对象，故返回自己
		pass

	def __next__(self):
		self.a ,self.b = self.b ,self.a + self.b
		if self.a >10000:
			raise StopIteration()
			pass
		return self.a

	def __getitem__(self,n):	#像list那样按照下标取出元素，需要实现__getitem__()方法
		a,b = 1,1
		for x in range(n):
			a,b = b, a+b
		return a 

for n in Fib():
	print(n)
	pass
