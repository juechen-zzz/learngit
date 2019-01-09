##继承和多态
class Animal(object):	
	def run(self):
		print('Animal is running')
		pass
	def run_twice(object):
		print('Animal is running too')
		pass

class Dog(Animal):		#继承父类Animal
	def run(self):
		print('Dog is running')
		pass
	pass

class Cat(Animal):
	def run(self):		#同样的定义函数cat会覆盖父类的同名函数
		print('Cat is running')
		pass
	pass

class Timer(object):	#不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了
    def run(self):
        print('Start...')