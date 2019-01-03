##面向对象，学生成绩
class student(object):
	def __init__(self,name,score):		#内部函数，self必不可少
		self.name = name
		self.score = score
		pass
	def print_score(self):
		print('%s：%s' % (self.name,self.score))
		pass
	def get_grade(self):
		if self.score > 90 :
			return 'A'
		elif self.score >80 :
			return 'B'
		elif self.score >60 :
			return 'C'
		else:
			return 'D'
		pass

#	bart = student('bart',59)
#	lisa = student('lisa',75)
#	bart.print_score
#	lisa.print_score