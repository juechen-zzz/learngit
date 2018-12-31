##求素数
def odd_iter():			#构造一个奇数序列
    n = 1
    while True:
        n = n + 2
        yield n

def not_divisible(n):		#加上筛选函数
    return lambda x: x % n > 0

def primes():
    yield 2
    it = odd_iter() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(not_divisible(n), it) # 构造新序列

# 打印1000以内的素数:
for n in primes():
    if n < 100:
        print(n)
    else:
        break