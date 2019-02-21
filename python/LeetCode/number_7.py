class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        a = abs(x)

        sum = 0
        while a > 0:
        	sum = sum * 10 + a % 10
        	a = a // 10
        	pass

        sum = sum if x >=0 else -sum

        return sum if sum < 2**31 and sum > -2**31 else 0