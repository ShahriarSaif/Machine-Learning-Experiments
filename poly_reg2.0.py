import numpy as np
import math
from matplotlib import pyplot as plt
poly_order = 4 #this is the optimal polynomial order

def generate_data():
	tr_x = np.arange(1, 60, 0.5)
	tr_y = [math.sin(0.1*x) for x in tr_x]
	tr_y = np.array(tr_y)
	test_x = np.arange(70, 90, 0.5)
	test_y = [math.sin(0.1*x) for x in test_x]
	test_y = np.array(test_y)
	tr_data = []
	test_data = []
	m = len(tr_x)
	for i in range(m):
		tr_data.append((tr_x[i], tr_y[i]))
	m = len(test_x)
	for i in range(m):
		test_data.append((test_x[i], test_y[i]))
	return (tr_data, test_data)

def fit_model(tr_data):
	w = np.random.random(poly_order + 1)
	X = []
	m = len(tr_data)
	for i in range(m):
		vec = []
		for j in range(poly_order + 1):
			vec.append(math.pow(tr_data[i][0], j))
		X.append(vec)
	X = np.array(X)
	X = np.linalg.pinv(X)
	y = []
	for i in range(m):
		y.append(tr_data[i][1])
	w = np.dot(X, y)
	return w

def model(x, w):
	a = []
	for i in range(poly_order + 1):
		a.append(math.pow(x, i))
	return np.dot(a, w)

def evaluate_model(data, w):
	sum = 0.0
	m = len(data)
	for i in range(m):
		sum += (math.pow(data[i][1] - model(data[i][0], w), 2))
	sum *= (1 / 2 * m)
	return sum	

def main():
	(tr_data, test_data) = generate_data()
	m_test = len(test_data)
	m_tr = len(tr_data)
	w = fit_model(tr_data)
	(tr_x, tr_y) = ([], [])
	(test_x, test_y) = ([], [])

	for i in range(m_test):
		test_x.append(test_data[i][0])
		test_y.append(test_data[i][1])
	for i in range(m_tr):
		tr_x.append(tr_data[i][0])
		tr_y.append(tr_data[i][1])

	error = evaluate_model(test_data, w)
	print('sum of square error = ', error)

	(fig, axs) = plt.subplots(2, 1, constrained_layout=True)
	axs[0].plot(test_x, test_y)
	axs[0].set_title('fitted model')
	axs[1].plot(tr_x, tr_y)
	axs[1].set_title('generated data')
	plt.show()

main()