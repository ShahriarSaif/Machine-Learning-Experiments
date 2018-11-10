import numpy as np
import math
import data_unwraper as du
portion = 20

class Perceptron:
	alpha = 0.3
	def __init__(self):
		w = []

	def neuron(self, x):
		if np.dot(self.w, x) >= 0:
			return 1
		else:
			return 0

	def train_perceptron(self, data):
		m = len(data)
		self.w = np.random.random(len(data[0][0]))
		for i in range(m):
			delta = data[i][1] - Perceptron.neuron(self, data[i][0])
			self.w = self.w + Perceptron.alpha * (delta * data[i][0])

def main():
	(tr_data, test_data) = du.unwrap_iris(portion)
	prcpt = Perceptron()
	m = len(test_data)
	prcpt.train_perceptron(tr_data)
	mc = 0
	for i in range(m):
		resp = prcpt.neuron(test_data[i][0])
		if resp != test_data[i][1]:
			mc += 1
	print('number of misclassification = ', mc)
	
main()