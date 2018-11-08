from matplotlib import pyplot as plt
import math
import numpy
import random as rnd

max_iter = 1000
alpha = 0.5

def f(x):
	return x * x

def df(x):
	return 2 * x

def gradient_descent(initial, tolerance):
	i = 1
	while i <= max_iter:
		x_min = initial - alpha * df(initial)
		if(abs(initial - x_min) <= tolerance):
			break
		i += 1
	return x_min

def main():
	x = numpy.arange(-50, 50, 0.5)
	y = [f(i) for i in x]
	x_min = gradient_descent(rnd.randint(5, 20), 1e-5)
	print('minimum point = ', x_min)
	plt.plot(x, y)
	plt.show()

main()