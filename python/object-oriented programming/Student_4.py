class Student(object):
	def __init__(self,name):
		self.name = name
		pass
	def __str__(self):
		return 'Student object (name:%s)' % self.name

	__repr__ = __str__	#再定义一个__repr__()。但是通常__str__()和__repr__()代码都是一样的