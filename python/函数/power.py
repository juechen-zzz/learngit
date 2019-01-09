# -*- coding: utf-8 -*-
##不变参数
##默认设置为n = 2
def power(x, n = 2):
	s = 1;
	while n > 0:
		n = n - 1;
		s = s * x;
	return s;