from sklearn.datasets import load_iris
import numpy as np
import math
k = 15

def prepare_data():
	data = load_iris()
	m = len(data.target)
	for i in range(m):
		if data.target[i] == 2:
			data.target[i] = 1
	return data

def distance(v1, v2):
	sum = math.sqrt(np.sum(np.power((v1 - v2), 2)))
	return sum

def k_nearest_neighbor(data, x, k):
	pair = []
	m = len(data)
	for i in range(m):
		pair.append((distance(x, data[i][0]), i))
	pair.sort()
	y = 0.0

	for i in range(k):
		indx = pair[i][1]
		y += data[indx][1]
	y *= 1 / k
	return k

def make_classification(data, x):
	resp = k_nearest_neighbor(data, x, k)
	if resp > 0.5:
		return 1
	return 0

def validate_accuracy():
	data = prepare_data()
	test_x = []
	test_y = []
	train_x = []
	train_y = []
	m = len(data.data)
	indx = np.random.permutation(m)
	test_x = data.data[indx[:20]]
	test_y = data.target[indx[:20]]
	train_x = data.data[indx[21:]]
	train_y = data.target[indx[21:]]
	test_data = []
	train_data = []
	m = len(test_x)
	for i in range(m):
		test_data.append((test_x[i], test_y[i]))
	m = len(train_x)
	for i in range(m):
		train_data.append((train_x[i], train_y[i]))

	m = len(test_data)
	mc = 0
	for i in range(m):
		resp = make_classification(train_data, test_data[i][0])
		if resp != test_data[i][1]:
			mc += 1
	print('number of misclassification = ', mc)

validate_accuracy()