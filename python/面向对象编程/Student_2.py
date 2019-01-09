class Student(object):
	def get_score(self):
		return self._score
		pass

	def set_score(self,value):		#设置score不能随便修改
		if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
		pass