import numpy
import math
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from matplotlib import pyplot

class NeuralNetwork:
	def __init__(self, net_arch):
		self.net_arch = net_arch
		self.n_train_layer = len(net_arch) - 1
		self.weights = []
		self.bias = []
		self.n_layer = len(net_arch)
		self.data = {}

		#initializing the weights and biasses
		for layer in range(self.n_layer-1):
			w = numpy.array(numpy.random.rand(net_arch[layer+1], net_arch[layer]))
			self.weights.append(w)
			b = numpy.array(numpy.random.rand(net_arch[layer+1]))
			self.bias.append(b)

	def custDot(self, a, b):
		res = []
		for i in range(len(a)):
			t = []
			for j in range(len(b)):
				t.append(a[i] * b[j])
			res.append(t)
		return numpy.array(res)


	def error(self):
		error = 0.0
		for i in range(len(self.data['data'])):
			e = self.data['target'][i] - self.predict(self.data['data'][i])
			error += e.dot(e)
		return (1.0 / len(self.data['data'])) * error

	def sigmoid(self, a):
		r = [self.sig(i) for i in a]
		return numpy.array(r)

	def sig(self, x):
		return 1.0 / (1.0 + numpy.exp(-x))

	def difSigmoid(self, a):
		r = [((1 - self.sig(i)) * self.sig(i)) for i in a]
		return numpy.array(r)

	def forwardProp(self, x):
		layer_out = [[] for i in range(self.n_layer)]
		layer_out[0] = x
		layer_out = numpy.array(layer_out)
		for layer in range(1, self.n_layer):
			a = self.sigmoid(self.weights[layer-1].dot(layer_out[layer-1]) + self.bias[layer-1])
			layer_out[layer] = a
		layer_out = numpy.array(layer_out)
		return layer_out

	def backProp(self, x, y):
		forward = self.forwardProp(x)
		error = numpy.array(y - forward[-1])
		sensitivities = [[] for i in range(self.n_train_layer)]
		f = numpy.diagflat(self.difSigmoid(forward[-1]))
		sensitivities[-1] = f.dot(error) * -2

		for layer in range(self.n_train_layer-1, 0, -1):
			f = numpy.diagflat(self.difSigmoid(forward[layer]))
			t = self.weights[layer].transpose().dot(sensitivities[layer])
			s = f.dot(t)
			sensitivities[layer-1] = s
		sensitivities = numpy.array(sensitivities)
		return sensitivities

	def predict(self, x):
		return self.forwardProp(x)[-1]

	def makeClassification(self, x):
		out = self.forwardProp(x)[-1]
		max = out[0]
		max_in = 0
		for i in range(len(out)):
			if max <= out[i]:
				max = out[i]
				max_in = i
		return max_in
			
	def fitNet(self, tr_data):
		self.data = tr_data
		current_error, previous_error = self.error(), float('inf')
		iter = 1
		tolerance = 0.3
		eps = 0.001

		while iter <= 2000: #math.fabs(self.error() - tolerance) > eps:
			for index in range(len(tr_data['data'])):
				sensitivities = self.backProp(tr_data['data'][index], tr_data['target'][index])
				a = self.forwardProp(tr_data['data'][index])
				for layer in range(self.n_layer-1, 0, -1):
					 self.weights[layer-1] = self.weights[layer-1] - ((1.0 / iter) * self.custDot(sensitivities[layer-1], a[layer-1]))
					 self.bias[layer-1] = self.bias[layer-1] - ((1.0 / iter) * sensitivities[layer-1])
			print('current error = ', self.error())
			print('iteration = ', iter)
			iter += 1
		print('converged')

def main():
	net = NeuralNetwork([4, 5, 3])
	data = load_iris()
	#data_new = {'data': normalize(data['data'], norm='l2', copy=True, return_norm=False), 'target': []}
	data_new = {'data': data['data'], 'target': []}

	for i in range(len(data['data'])):
		if data['target'][i] == 0:
			data_new['target'].append(numpy.array([1, 0, 1]))
		elif data['target'][i] == 1:
			data_new['target'].append(numpy.array([0, 1, 0]))
		else:
			data_new['target'].append(numpy.array([0, 0, 1]))

	n_mc = 0
	net.fitNet(data_new)
	for i in range(len(data_new['data'])):
		act = data['target'][i]
		print('actual class = ', act)
		resp = net.makeClassification(data_new['data'][i])
		print('network response = ', resp)
		if resp != act:
			n_mc += 1
	print('number of misclassification = ', n_mc)

main()