##排序
a = [1,2,3,4,-9,-6];

b = sorted(a);									#正常排序
print(b);

c = sorted(a,key = abs);						#设置为绝对值排序
print(c);

d =['bob', 'about', 'Zoo', 'Credit'];

e = sorted(d);									#字符串正常排序
print(e);

f = sorted(d,key = str.lower);					#设置忽略大小写
print(f);

g =sorted(d,key = str.lower,reverse=True);		#设置反向排序
print(g);