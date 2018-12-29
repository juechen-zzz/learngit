##参数组合
def f1(a,b,c=0,*args,**kw):
	 print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw);

def f2(a, b, c=0, *, d, **kw):
	 print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw);


