# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        node1 = l1
        node2 = l2
        l3 = ListNode(0)
        l3.next = ListNode(0)#[0],[0]的特殊情况
        node3 = l3
        
        sum1 = 0
        coe = 1
        while not node1 is None:
            sum1 += node1.val * coe
            coe *= 10
            node1 = node1.next
        
        sum2 = 0
        coe =1
        while not node2 is None:
            sum2 += node2.val * coe
            coe *= 10
            node2 = node2.next
        
        sum = sum1 + sum2
        
        while sum > 0:
            node3.next = ListNode(sum % 10)
            node3 = node3.next
            sum //= 10
            
        return l3.next    