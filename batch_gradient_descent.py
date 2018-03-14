import sys
import time
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn import preprocessing
import matplotlib.pyplot as plt


class GradientDescentManager:

	def __init__(self, X, Y):

		self.raw_data = [X, Y]
		self.data = [np.array(X), np.array(Y)]
		self.add_features()
		self.max_values = np.max(self.data[0], 0)[1:]
		self.min_values = np.min(self.data[0], 0)[1:]
		self.normalize()
		self.m = len(X)
		self.N = len(X[0])

		self.alpha = 0.01
		self.theta = []
		self.steps = 1
		self.thetas = np.array([])
		self.costs = []

	def normalize(self, to_add=False):
		if not to_add:
			for idx, example in enumerate(self.data[0]):
				self.data[0][idx][1:] = (example[1:] - self.min_values) / (self.max_values - self.min_values)
		else:
			return [np.concatenate([[1], (i[1:] - self.min_values) / (self.max_values - self.min_values)]) for i in to_add]

	def add_features(self, to_add=False):
		if not to_add:
			self.data[0] = [np.concatenate([[1], x, x**2, x**.5]) for x in self.data[0]]
		else:
			print('to_add',to_add)
			return np.concatenate([[1], to_add, to_add**2, to_add**.5])

	def compute_cost(self):
		cost = 0
		for i in range(self.m):
			cost += (self.data[0][i].dot(self.theta) - self.data[1][i]) ** 2
		return 1 / (2*self.m) * cost

	def get_gradient(self):
		gradients = np.zeros(len(self.theta))
		for i in range(self.m):
			gradients += (self.data[0][i].dot(self.theta) - self.data[1][i]) * self.data[0][i]
		return 1 / self.m * gradients

	def run(self, steps):
		start_time = time.time()

		for step in range(steps):
			self.steps += 1

			C = self.compute_cost()
			grad = self.get_gradient()

			self.theta -= self.alpha * grad
			self.costs.append(C)
			self.thetas = np.vstack((self.thetas, self.theta))

		total_time = time.time() - start_time
		print(str(steps) + ' steps done in ' + str(total_time))

	def create_theta(self):
		self.theta = np.random.random((len(self.data[0][0]))) * 10 - 5
		self.thetas = [self.theta]

	def plot_data(self):
		#if data is 1 dimensional 
		if self.N == 1:
			#plot raw data
			plt.plot([i[0] for i in self.raw_data[0]], self.raw_data[1], 'bo')

			theta_plots = 5
			colors = ['firebrick', 'red', 'darkorange', 'gold', 'greenyellow']
			theta_plot_accuracy = 20

			x_min = min(self.raw_data[0])[0]
			x_max = max(self.raw_data[0])[0]

			x_step = (x_max - x_min)/theta_plot_accuracy
			theta_step = int(len(self.thetas)/(theta_plots+1))

			X_to_plot = [np.array([x_min + i*x_step]) for i in range(theta_plot_accuracy+1)]
			X_modified = self.normalize([self.add_features(i) for i in X_to_plot])

			for j in range(theta_plots):
				H_to_plot = [self.thetas[j*theta_step].dot(i) for i in X_modified]
				plt.plot(X_to_plot, H_to_plot, colors[j], label='theta after ' + str(j*theta_step) + 'iterations')

			j = theta_plots

			X_to_plot = [np.array([x_min + i*x_step]) for i in range(theta_plot_accuracy+1)]
			X_modified = self.normalize([self.add_features(i) for i in X_to_plot])
			H_to_plot = [self.thetas[j*theta_step].dot(i) for i in X_modified]

			plt.plot(X_to_plot, H_to_plot, 'g', label='final theta', linewidth=3.0)
			plt.legend(loc='upper left')
			plt.ylabel('Y')
			plt.xlabel('X')
			plt.show()

		#if data is 2 dimensional
		elif self.N == 2:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			X1 = [i[0] for i in self.raw_data[0]]
			X2 = [i[1] for i in self.raw_data[0]]
			Y = self.raw_data[1]
			ax.scatter(X1, X2, Y)

			mn = np.amin(self.raw_data[0], axis=0)
			mx = np.amax(self.raw_data[0], axis=1)
			
			x1_min = mn[0]
			x1_max = mx[0]
			x2_min = mn[1]
			x2_max = mx[1]

			theta_plot_accuracy = 20

			X1_to_plot = np.linspace(x1_min, x1_max, num=theta_plot_accuracy)
			X2_to_plot = np.linspace(x2_min, x2_max, num=theta_plot_accuracy)

			''' not working yet
			X1_modified = self.normalize([self.add_features(np.array([i])) for i in X1_to_plot])
			X2_modified = self.normalize([self.add_features(np.array([i])) for i in X2_to_plot])

			X1, X2 = np.meshgrid(X1_to_plot, X2_to_plot)

			H_to_plot = np.zeros((X1.shape))
			for h in range(len(X1_modified)):
				for w in range(len(X2_modified)):
					print(H_to_plot[h][w])
					print(self.theta.dot([X1_modified[h], X2_modified[w]]))
					H_to_plot[h][w] = self.theta.dot([X1[h][w], X2[h][w]])
			'''

			ax.set_xlabel('X1 axis')
			ax.set_ylabel('X2 axis')
			ax.set_zlabel('Y axis')
			plt.show()

	def plot_costs(self):
		plt.plot([i for i in range(len(self.costs))], self.costs)
		plt.ylabel('Cost')
		plt.xlabel('Number of iterations')
		plt.show()

file = open('ex1data2.txt')
content = file.read().splitlines()
file.close()

X = [np.array(i.split(',')[:-1], dtype=float) for i in content]
Y = [np.array(i.split(',')[-1], dtype=float) for i in content]

manager = GradientDescentManager(X, Y)
manager.create_theta()

print('starting cost:', manager.compute_cost())
print('starting theta:', manager.theta)

manager.run(500)

print('final cost:', manager.compute_cost())
print('final theta:', manager.theta)

manager.plot_data()
manager.plot_costs()