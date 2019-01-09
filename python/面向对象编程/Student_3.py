##运用 @property
class Student(object):
    @property  			#标记只读
    def score(self):
        return self._score

    @score.setter		#标记可写
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value