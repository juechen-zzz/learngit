##匿名函数lambda
l = list(map(lambda x : x*x , [1,2,3,4]));
print(l);

f = lambda s:s*s;
print(f(4));

s=list(filter(lambda n:n%2==1,range(1,20)))
print(s);