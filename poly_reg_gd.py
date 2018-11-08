import numpy
import math
from matplotlib import pyplot as plt
import random

poly_order = 10
alpha = 0.5
max_iter = 1000

def generate_data():
	x = numpy.arange(0, 50, 0.5)
	y = [math.sin(0.1*i) for i in x]
	x = numpy.array(x)
	y = numpy.array(y)
	return (x, y)

def prepare_feature(x):
	feature_map = []
	for i in x:
		t = []
		for j in range(poly_order + 1):
			t.append(math.pow(i, j))
		feature_map.append(t)
	feature_map = numpy.array(feature_map)
	return feature_map

def model(x, w):
	return x.dot(w)

def error(feature_map, y, w):
	sum = 0.0
	m = len(feature_map)
	for i in range(m):
		sum += (y[i] - model(feature_map[i], w)) ** 2
	sum *= (1 / 2 * m)
	return sum

def d_error(x, y, w):
	sum = 0.0
	m = len(x)
	n = len(x[0])

	for i in range(m):
		for j in range(n):
			sum += (y[i] - model(x[i], w)) * x[i][j]
	sum *= -(1 / m)
	return sum

def gradient_descent(x, y):
	w = []
	for i in range(poly_order + 1):
		w.append(random.random())
	w = numpy.array(w)
	tolerance = 1e-5
	i = 1
	while i <= max_iter:
		w_new = w - (alpha * d_error(x, y, w))
		w_new = numpy.array(w_new)
		if numpy.sum(numpy.abs(w_new - w)) <= tolerance:
			break
		i += 1
	return w_new


(x, y) = generate_data()
feature_map = prepare_feature(x)
w = gradient_descent(feature_map, y)
print(w)