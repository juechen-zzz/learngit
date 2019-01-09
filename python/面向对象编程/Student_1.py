class Student(object):
	def set_age(self,age):
		self.age = age
		pass
	__slots__ = ('name', 'age')	# 用tuple定义允许绑定的属性名称