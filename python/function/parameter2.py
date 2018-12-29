##可变参数
def calc(numbers):
	sum = 0;
	for n in numbers:
		sum = sum + n * n;
	return sum;

##输入必须是list或者tuple，可改为可变参数
def calc2(*numbers):
	sum = 0;
	for n in numbers:
		sum = sum + n * n;
	return sum;

##关键字参数
def person(name,age,**kw):
	print('name:',name,'age:',age,'other:',kw);

##限制关键字参数的名字city job
def person2(name, age, *, city, job):
    print(name, age, city, job)

##
def person3(name, age, *args, city, job):
    print(name, age, args, city, job)

##可以设定默认值
def person4(name, age, *, city='Beijing', job):
    print(name, age, city, job);