def _private_1(name):
	return 'hello,%s' % name
	pass

def _private_2(name):
	return 'hi,%s' % name
	pass

def greeting(name):
	if(len(name)>3):
		return _private_1(name)
	else:
		return _private_2(name)
	pass