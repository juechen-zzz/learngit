###dict查找
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85};
print(d['Michael']);
#增加一个key
d['Adam'] = 67;
print(d['Adam']);
#删除一个key
d.pop('Bob');
print(d);


###set查找
s = set([1,1,2,3]);
print(s);
#增加
s.add(4);
print(s);
#删除
s.remove(2);
print(s);
