def fib(n):
	if n < 1:
		print('输入有误！')
		return -1
	elif n == 1 or n == 2:
		return 1
	else:
		return fib(n-1) + fib(n-2)

number = int(input('输入一个整数书：'))
result = fib(number)
print(result)