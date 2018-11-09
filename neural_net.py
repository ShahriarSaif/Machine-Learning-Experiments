import numpy as np
import math
from sklearn.datasets import load_iris
import random as rnd

n_in = 5
n_hid = 5
n_out = 1
alpha = 0.3

def prepare_data():
	data = load_iris()
	portion = 20
	indx = np.random.permutation(len(data))
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	train_x = data.data[indx[:portion]]
	train_y = data.target[indx[:portion]]
	test_x = data.data[indx[portion:]]
	test_y = data.target[indx[:portion]]
	m_train = len(train_x)
	m_test = len(test_x)
	test_data = []
	train_data = []

	for i in range(m_train):
		np.append(train_x[i], 1)
		train_data.append((train_x[i], train_y[i]))

	for i in range(m_test):
		np.append(test_x[i], 1)
		test_data.append((test_x[i], test_y[i]))

	return (train_data, test_data)

def sigmoid(x):
	return 1 / 1 + math.exp(-x)

def neuron(w, x):
	return sigmoid(numpt.dot(w, x))

def fit_net(tr_data):
	in_layer_out = []
	hid_layer_out = []
	out_layer_out = []
	hid_layer_error = []
	out_layer_error = []
	w_in2hid = np.random.rand(n_hid, n_in)
	w_hid2out = np.random.rand(n_out, n_hid)
	m = len(tr_data)
	i = 0
	while i < m:
		in_layer_out = tr_data[i][0]
		hid_layer_out = [neuron(w_in2hid[j], in_layer_out) for j in range(n_hid)]
		out_layer_out = [neuron(w_hid2out[k], hid_layer_out) for k in range(n_out)]
		for k in range(n_out):
			a = out_layer_out[k] * (1 - out_layer_out[k]) * (out_layer_out[k] - tr_data[i][1][k])
			out_layer_error.append(a)

		for j in range(n_hid):
			sum = 0
			for k in range(n_out):
				sum += out_layer_error[k] * w_hid2out[j][k]
			a = hid_layer_out[j] * (1 - hid_layer_out[j]) * sum
			hid_layer_error.append(a)

		for j in range(n_hid):
			for k in range(n_in):
				w_in2hid[j][k] -= alpha * hid_layer_error[j] * in_layer_out[k]
		for j in range(n_out):
			for k in range(n_hid):
				w_hid2out[j][k] -= alpha * out_layer_error[j] * hid_layer_out[k]
		i += 1
	return (w_in2hid, w_hid2out)

def make_classification(data, x):
	(w_in2hid, w_hid2out) = fit_net(data)
	hid_layer_out = [neuron(w_in2hid[j], x) for j in range(n_hid)]
	out_layer_out = [neuron(w_hid2out[k], hid_layer_out) for k in range(n_out)]
	return out_layer_out

def main():
	(train_data, test_data) = prepare_data()
	mc = 0
	m = len(test_data)
	for i in range(m):
		resp = make_classification(train_data, test[i][0])
		if resp != test_data[i][1]:
			mc += 1
	print('number of misclassification = ', mc)

main()