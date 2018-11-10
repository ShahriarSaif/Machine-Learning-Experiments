import numpy as np
import math
from sklearn.datasets import load_iris
import random as rnd

n_in = 5
n_hid = 5
n_out = 1
alpha = 0.3
max_iter = 500

w_in2hid = []
w_hid2out = []

def prepare_data():
	data = load_iris()
	m = len(data.data)
	portion = 20
	indx = np.random.permutation(m)
	test_x = data.data[indx[:portion]]
	test_y = data.target[indx[:portion]]
	tr_x = data.data[indx[portion:]]
	tr_y = data.target[indx[portion:]]
	tr_data = []
	test_data = []
	m = len(tr_x)
	for i in range(m):
		a = np.append(tr_x[i], 1)
		if tr_y[i] == 2:
			tr_y[i] = 1
		tr_data.append((a, tr_y[i]))
	m = len(test_x)
	for i in range(m):
		a = np.append(test_x[i], 1)
		if test_y[i] == 2:
			test_y[i] = 1
		test_data.append((a, test_y[i]))
	return (tr_data, test_data)

def sigmoid(x):
	return 1 / 1 + math.exp(-x)

def neuron(w, x):
	return sigmoid(np.dot(w, x))

def fit_net(tr_data):
	in_layer_out = []
	hid_layer_out = []
	out_layer_out = []
	hid_layer_error = []
	out_layer_error = []
	global w_in2hid
	global w_hid2out
	w_in2hid = np.random.rand(n_hid, n_in)
	w_hid2out = np.random.rand(n_out, n_hid)
	m = len(tr_data)
	iteration = 1

	while iteration <= max_iter:
		i = 0
		while i < m:
			in_layer_out = tr_data[i][0]
			hid_layer_out = [neuron(w_in2hid[j], in_layer_out) for j in range(n_hid)]
			out_layer_out = [neuron(w_hid2out[k], hid_layer_out) for k in range(n_out)]
			for k in range(n_out):
				a = out_layer_out[k] * (1 - out_layer_out[k]) * (out_layer_out[k] - tr_data[i][1])
				out_layer_error.append(a)

			for j in range(n_hid):
				sum = 0
				for k in range(n_out):
					sum += out_layer_error[k] * w_hid2out[k][j]
				a = hid_layer_out[j] * (1 - hid_layer_out[j]) * sum
				hid_layer_error.append(a)

			for j in range(n_hid):
				for k in range(n_in):
					w_in2hid[j][k] -= alpha * hid_layer_error[j] * in_layer_out[k]
			for j in range(n_out):
				for k in range(n_hid):
					w_hid2out[j][k] -= alpha * out_layer_error[j] * hid_layer_out[k]
			i += 1
		iteration += 1

def make_classification(x):
	global w_in2hid
	global w_hid2out
	hid_layer_out = [neuron(w_in2hid[j], x) for j in range(n_hid)]
	out_layer_out = [neuron(w_hid2out[k], hid_layer_out) for k in range(n_out)]
	return out_layer_out

def main():
	(train_data, test_data) = prepare_data()
	mc = 0
	m = len(test_data)
	fit_net(train_data)
	for i in range(m):
		resp = make_classification(test_data[i][0])
		print('net response = ', resp[0])
		print('actual response = ', test_data[i][1])
		if resp[0] != test_data[i][1]:
			mc += 1
	print('number of misclassification = ', mc)

main()