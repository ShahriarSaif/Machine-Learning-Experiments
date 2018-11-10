import numpy as np
import math
import random as rnd
from matplotlib import pyplot as plt

alpha = 0.3
max_iter = 1000
poly_order = 3
tolerance = 1e-5

def prepare_features(x):
	res = []
	for i in range(poly_order + 1):
		res.append(math.pow(x, i))
	return res

def generate_data():
	tr_x = np.arange(1, 60, 0.5)
	tr_y = [math.sin(0.1*x) for x in tr_x]
	tr_y = np.array(tr_y)
	m = len(tr_x)
	tr_data = []
	for i in range(m):
		tr_x_feature = []
		for j in range(poly_order + 1):
			tr_x_feature.append(math.pow(tr_x[i], j))
		tr_data.append((np.array(tr_x_feature), tr_y[i]))

	test_x = np.arange(70, 90, 0.5)
	test_y = [math.sin(0.1*x) for x in test_x]
	test_y = np.array(test_y)
	test_data = []
	m = len(test_x)
	for i in range(m):
		test_x_feature = []
		for j in range(poly_order + 1):
			test_x_feature.append(math.pow(test_x[i], j))
		test_data.append((np.array(test_x_feature), test_y[i]))
	return (tr_data, test_data)

def model(x, w):
	return np.dot(x, w)

def cost(data, w):
	sum = 0.0
	m = len(data)
	for i in range(m):
		a = model(data[i][0], w) - data[i][1]
		sum += math.pow(a, 2)
	sum *= (1 / 2 * m)
	return sum

def dcost(data, w):
	sum = np.zeros(poly_order + 1)
	m = len(data)
	for i in range(m):
		a = (model(data[i][0], w) - data[i][1]) * data[i][0] 
		sum = sum + a
	sum = sum * -(1 / m)
	return sum

def fit_model(data):
	w = np.random.random(poly_order + 1)
	iter = 1
	while iter <= max_iter:
		w_new = w - (alpha * dcost(data, w))
		if np.sum(np.abs(w - w_new)) <= tolerance:
			break
		iter += 1
	return w_new

def main():
	(tr_data, test_data) = generate_data()
	m_test = len(test_data)
	m_tr = len(tr_data)
	w = fit_model(tr_data)
	error = cost(test_data, w)
	print('sum of square error = ', error)
	x = np.arange(0, 100, 0.5)
	y = [math.sin(0.1*i) for i in x]
	x_fitted = x
	x_fitted_mat = []
	y_fitted = []
	for i in x:
		x_fitted_mat.append(prepare_features(i))
	for i in x_fitted_mat:
		y_fitted.append(model(i,  w))
	
	(fig, axs) = plt.subplots(2, 1, constrained_layout=True)
	axs[0].plot(x_fitted, y_fitted)
	axs[0].set_title('fitted model')
	axs[1].plot(x, y)
	axs[1].set_title('generated data')
	plt.show()
main()

	