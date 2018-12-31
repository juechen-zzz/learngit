##构造回数，左右读一样
def is_palindrome(n):
	return str(n)==str(n)[::-1]
output=filter(is_palindrome, range(1,101))
print(list(output))