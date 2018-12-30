##杨辉三角
def triangles():
    L=[1]                                                         
    while True:
        yield L                                                           #打印出该list
        L=[L[x]+L[x+1] for x in range(len(L)-1)]       					 #计算下一行中间的值
        L.insert(0,1)                                          	       #在开头插入1
        L.append(1)                                         	        #在结尾添加1
        if len(L)>10:                                        	         #仅输出10行
            break