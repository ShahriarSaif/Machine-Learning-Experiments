import numpy as np
import math
from sklearn.datasets import load_iris
import random as rnd

def prepare_data(portion):
    data = load_iris()
    tr_data = []
    test_data = []



class NeuralNetwork:
    n_in = 3
    n_hid = 3
    n_out = 1

    def __init__(self):
        self.w_in2hid = []
        self.w_hid2out = []

    def sigmoid(self, x):
        return 1 / 1 + math.exp(-x)

    def neuron(self, w, x):
        return self.sigmoid(np.dot(w, x))

    def fit_net(self, alpha, max_iter, tr_data):
	       in_layer_out = []
	       hid_layer_out = []
	       out_layer_out = []
	       hid_layer_error = []
	       out_layer_error = []
	       self.w_in2hid = np.random.rand(NeuralNetwork.n_hid, NeuralNetwork.n_in)
	       self.w_hid2out = np.random.rand(NeuralNetwork.n_out, NeuralNetwork.n_hid)
	       m = len(tr_data)

	       iteration = 1
	       while iteration <= max_iter:
                i = np.random.randint(0, m)
                in_layer_out = tr_data[i][0]
                hid_layer_out = [self.neuron(self.w_in2hid[j], in_layer_out) for j in range(NeuralNetwork.n_hid)]
                out_layer_out = [self.neuron(self.w_hid2out[k], hid_layer_out) for k in range(NeuralNetwork.n_out)]
                for k in range(NeuralNetwork.n_out):
                    a = out_layer_out[k] * (1 - out_layer_out[k]) * (out_layer_out[k] - tr_data[i][1])
                    out_layer_error.append(a)
                for j in range(NeuralNetwork.n_hid):
                    sum = 0
                    for k in range(NeuralNetwork.n_out):
                        sum += out_layer_error[k] * self.w_hid2out[k][j]
                    a = hid_layer_out[j] * (1 - hid_layer_out[j]) * sum
                    hid_layer_error.append(a)
                for j in range(NeuralNetwork.n_hid):
                    for k in range(NeuralNetwork.n_in):
                        self.w_in2hid[j][k] -= alpha * hid_layer_error[j] * in_layer_out[k]
                for j in range(NeuralNetwork.n_out):
                    for k in range(NeuralNetwork.n_hid):
                        self.w_hid2out[j][k] -= alpha * out_layer_error[j] * hid_layer_out[k]
                iteration += 1

    def make_classification(self, x):
	       hid_layer_out = [self.neuron(self.w_in2hid[j], x) for j in range(NeuralNetwork.n_hid)]
	       out_layer_out = [self.neuron(self.w_hid2out[k], hid_layer_out) for k in range(NeuralNetwork.n_out)]
	       return out_layer_out

data = [[[0, 0, 1], 0], [[0, 1, 1], 1], [[1, 0, 1], 1], [[1, 1, 1], 1]]
net = NeuralNetwork()
net.fit_net(0.5, 100, data)
print(net.make_classification(data[0][0]))
