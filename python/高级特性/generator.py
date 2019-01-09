#生成器
L = [x * x for x in range(10)];
print(L);
g = (x * x for x in range(10));
print(g);

#下一个值
print(next(g));

#循环输出
for n in g:
	print(n);
	pass