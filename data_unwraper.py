from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
import numpy as np

def unwrap_iris(portion):
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

def unwrap_boston(portion):
	data = load_boston()
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

def unwrap_diabetes(portion):
	data = load_diabetes()
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

def unwrap_linnerud(portion):
	data = load_linnerud()
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

def unwrap_wine(portion):
	data = load_wine()
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

def unwrap_breast_cancer(portion):
	data = load_breast_cancer()
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