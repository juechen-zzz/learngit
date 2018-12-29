##不变参数
##错误编法
def add_1 (l = []):
	l.append('end');
	return l;

##正确
def add_2(l = None):
	if l is None:
		l = [];
	l.append('end');
	return l ;