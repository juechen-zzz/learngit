from collections import deque


a = deque()
a.append(1)
a.append(2)
a.append(3)
print(a)

a.appendleft(4)
print(a)

a.pop()
print(a)

a.popleft()
print(a)

