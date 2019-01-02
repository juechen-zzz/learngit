# -*- coding: utf-8 -*-

' a test module'	#一个字符串，表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释
__author__ ='nihaopeng' 	#使用__author__变量把作者写进去`

import sys

def test():
	args = sys.argv
	if(len(args)==1):
		print('hello world')
	elif(len(args)==2):
		print('hello ,%s!' %args[1])
	else:
		print('too many arguments')
	pass

if __name__ == '__main__':
	test()
