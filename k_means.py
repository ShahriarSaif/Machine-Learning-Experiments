import numpy as np
import math
import random as rnd
from matplotlib import pyplot as plt

n_iter = 10
m = 100

def generate_data(m):
	data_mat = []
	for i in range(m):
		x = rnd.randint(0, 50)
		y = rnd.randint(0, 50)
		data_mat.append([x, y])
	data_mat = np.array(data_mat)
	return data_mat

def distance(v1, v2):
	d = len(v1)
	sum = 0.0
	for i in range(d):
		sum += math.pow(v2[i] - v1[i], 2)
	d = math.sqrt(sum)
	return d

def vect_avg(mat):
	m = len(mat[0])
	sum = np.zeros(m)
	for i in range(m):
		sum = sum + mat[i]
	sum *= 1 / m
	return sum

def k_means_cluster(data):
	m = len(data)
	n = len(data[0])
	cluster_centroid1 = data[rnd.randint(0, m)]
	cluster_centroid2 = data[rnd.randint(0, m)]
	j = 1

	while j <= n_iter:
		cluster1_data = []
		clusetr2_data = []
		for i in range(m):
			if(distance(cluster_centroid1, data[i]) < distance(cluster_centroid2, data[i])):
				cluster1_data.append(data[i])
			else:
				clusetr2_data.append(data[i])
		cluster_centroid1 = vect_avg(cluster1_data)
		cluster_centroid2 = vect_avg(clusetr2_data)
		j += 1
	return (cluster1_data, clusetr2_data)


def main():
	data = generate_data(m)
	(cd1, cd2) = k_means_cluster(data)
	x1 = []
	x2 = []

	for i in range(m):
		x1.append(data[i][0])
		x2.append(data[i][1])
	c1_x1 = []
	c1_x2 = []
	c2_x1 = []
	c2_x2 = []

	m_cd1 = len(cd1)
	for i in range(m_cd1):
		c1_x1.append(cd1[i][0])
		c1_x2.append(cd1[i][1])
	m_cd2 = len(cd2)
	for i in range(m_cd2):
		c2_x1.append(cd2[i][0])
		c2_x2.append(cd2[i][1])

	(fig, axs) = plt.subplots(3, 1, constrained_layout=True)
	axs[0].scatter(x1, x2)
	axs[0].set_title('data')
	axs[1].scatter(c1_x1, c1_x2, color = 'red')
	axs[1].set_title('cluster 1 data')
	axs[2].scatter(c2_x1, c2_x2, color = 'green')
	axs[1].set_title('cluster 2 data')
	plt.show()

main()