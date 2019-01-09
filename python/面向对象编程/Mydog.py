class Mydog(object):
	def __len__(self):
		return 100
		pass


class Myobject(object):
	def __init__(self):
		self.x = 9
	def power(self):
		return self.x * self.x
		pass

# hasattr(obj, 'x') # 有属性'x'吗？
# obj.x
# hasattr(obj, 'y') # 有属性'y'吗？
# setattr(obj, 'y', 19) # 设置一个属性'y'
# hasattr(obj, 'y') # 有属性'y'吗？
# getattr(obj, 'y') # 获取属性'y'
# obj.y # 获取属性'y'
