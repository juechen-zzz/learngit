# -*- coding: utf-8 -*-
##一元二次方程求解
import math;
def quadratic(a,b,c):
	m = b*b - 4*a*c;
	x1 = (-b + math.sqrt(m))/(-2*a);
	x2 = (-b - math.sqrt(m))/(-2*a);
	return x1,x2

r = quadratic(1,6,9);
print(r);