##偏函数
import functools
a = int ('40',base = 16);			#进制转换
print(a);

int2 = functools.partial(int, base=2);
print(int2('10010'));