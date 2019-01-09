##定义移动步骤
import math;
def move(x,y,step,angle = 0):
	nx = x + step*math.cos(angle);
	ny = y - step*math.sin(angle);
	return nx,ny

r = move(100, 100, 60, math.pi / 6);
print(r);