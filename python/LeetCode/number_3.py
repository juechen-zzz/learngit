class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = 0 
        max_length = 0
        substring = {}
        for i, c in enumerate(s):
            if c in substring and start <= substring[c]:
                start = substring[c] + 1
            else:
                max_length = max(max_length, i - start + 1)
            substring[c] = i
            
        return max_length