##切片
l = ['a','b','c','d'];
print(l[0:3]);
print(l[1:3]);
print(l[:3]);
print(l[-2:]);

#前10个数，每两个取一个
s = list(range(20));
print(s[:10:2]);

#每五个取一个
print(s[::5]);

#复制一个list
print(s[:]);

#tuple切片
t = ['a','b','c','d'];
print(t[:]);

#字符串切片
a = 'abcdef';
print(a[:]);