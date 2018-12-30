# -*- coding: utf-8 -*-
##去除字符串首尾的空格
def trim(s):
	if 0==len(s):
 		return s
      
	while ' '==s[0]:
		s=s[1:]
		if 0==len(s):
			return s
         
	while ' '==s[-1]:
		s=s[:-1]
		if 0==len(s):
			return s
	return s