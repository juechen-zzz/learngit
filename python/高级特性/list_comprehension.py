##列表生成式
#平方数
s = [x * x for x in range(1,10)];
print(s);

#加if语句
m =[x * x for x in range(1,10) if x % 2 ==0]
print(m);

#双层循环
n = [a + b for a in 'abc' for b in '123'];
print(n);

#双变量
d = {'x': 'A', 'y': 'B', 'z': 'C' };
r =[k + '=' + v for k, v in d.items()];
print(r);

#改小写
L = ['Hello', 'World', 'IBM', 'Apple'];
print([s.lower() for s in L]);