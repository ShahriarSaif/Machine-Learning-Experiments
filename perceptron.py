import numpy as np
import math
from sklearn.datasets import load_iris

alpha = 0.3
portion = 20

def prepare_data():
	tr_x = []
	tr_y = []
	test_x = []
	test_y = []
	data = load_iris()
	m = len(data)
	indx = np.random.permutation(m)
	tr_x = data.data[indx[:portion]]
	tr_y = data.target[indx[:portion]]
	test_x = data.data[indx[portion:]]
	test_y = dat.target[indx[portion:]]
	tr_data = []
	test_data = []
	
	m = len(tr_x)
	for i in range(m):
		tr_x[i].append(1)
		if tr_y[i] == 2:
			tr_y[i] = 1
		tr_data.append((tr_x[i], tr_y[i]))
	m = len(test_x)
	for i in range(m):
		test_x.append(1)
		if test_y[i] == 2:
			test_y[i] = 1
		test_data.append((test_x[i], test_y[i]))
	return (tr_data, test_data)

def perceptron(w, x):
	if np.dot(w, x) >= 0:
		return 1
	else:
		return 0

def train_perceptron(data):
	m = len(data)
	w = np.random.random(len(data[0][0]))
	for i in range(m):
		delta = data[i][1] - perceptron(w, data[i][0])
		w = w + alpha * (delta * data[i][0])
	return w

def main():
	(tr_data, test_data) = prepare_data()
	w = train_perceptron(tr_data)
	mc = 0
	m = len(test_data)
	for i in range(m):
		resp = perceptron(w, test_data[i][0])
		if resp != test_data[i][1]:
			mc += 1
	print('number of misclassification = ', mc)
	
main() 