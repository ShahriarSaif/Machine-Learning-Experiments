import numpy
from numpy import linalg
import math
from matplotlib import pyplot as plt
import random
poly_order = 4 #this is the optimal polynomial order

def generate_data():
	x = numpy.arange(0, 50, 0.5)
	y = [math.sin(0.1*i) for i in x]
	x = numpy.array(x)
	y = numpy.array(y)
	return (x, y)

def fit_model(x, y, m):
	w = []
	for i in range(poly_order + 1):
		w.append(random.random())
	X = []
	for i in range(m):
		vec = []
		for j in range(poly_order + 1):
			vec.append(math.pow(x[i], j))
		X.append(vec)
	X = numpy.array(X)
	X = numpy.linalg.pinv(X)
	w = numpy.dot(X, y)
	return w

def test_model(x, w):
	sum = 0.0
	for i in range(poly_order + 1):
		sum += w[i] * (x ** i)
	return sum

def evaluate_model(x, m, w):
	sum = 0.0
	for i in range(m):
		sum += (1 / 2*m) * ((test_model(x[i], w) - math.sin(0.1*x[i])) ** 2)
	return sum

def main():
	(tr_x, tr_y) = generate_data()
	m_tr = len(tr_x)
	w = fit_model(tr_x, tr_y, m_tr)
	tst_x = numpy.arange(20, 50, 0.5)
	tst_y = [test_model(i, w) for i in tst_x]
	m_tst = len(tst_x)

	error = evaluate_model(tst_x, m_tst, w)
	print('sum of square error = ', error)

	(fig, axs) = plt.subplots(2, 1, constrained_layout=True)
	axs[0].plot(tst_x, tst_y)
	axs[0].set_title('fitted model')
	axs[1].plot(tr_x, tr_y)
	axs[1].set_title('generated data')
	plt.show()

main()
