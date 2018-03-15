import sys
import time
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn import preprocessing
import matplotlib.pyplot as plt


class GradientDescentManager:

	def __init__(self, X, Y, alpha=0.1, polynomials=5):

		self.raw_data = [X, Y]
		self.data = [np.array(X), np.array(Y)]
		self.polynomials = polynomials
		self.poly = [i for i in range(2, self.polynomials + 2)]
		self.poly = np.concatenate((self.poly, [1/i for i in range(2, min(4, self.polynomials + 2))]))
		self.add_features()
		self.max_values = np.max(self.data[0], 0)[1:]
		self.min_values = np.min(self.data[0], 0)[1:]
		self.normalize()

		self.m = len(X)
		self.N = len(X[0])
		self.alpha = alpha
		self.theta = []
		self.steps = 1
		self.thetas = np.array([])
		self.costs = []

	def normalize(self, to_add=False, idx='False'):
		if not to_add:
			for idx, example in enumerate(self.data[0]):
				self.data[0][idx][1:] = (example[1:] - self.min_values) / (self.max_values - self.min_values)
		elif idx != 'False':
			l = int(len(self.min_values)/2)
			mn = np.array([self.min_values[idx + i*2] for i in range(l)])
			mx = np.array([self.max_values[idx + i*2] for i in range(l)])
			return [np.concatenate([[1], (i[1:] - mn) / (mx - mn)]) for i in to_add]
		else:
			return [np.concatenate([[1], (i[1:] - self.min_values) / (self.max_values - self.min_values)]) for i in to_add]

	def add_features(self, to_add=False):
		if not to_add:
			self.data[0] = [np.concatenate([[1], x, np.concatenate([x**p for p in self.poly])]) for x in self.data[0]]
		else:
			return np.concatenate([[1], to_add, np.concatenate([to_add**p for p in self.poly])])

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

			H_to_plot = [self.thetas[0].dot(i) for i in X_modified]
			plt.plot(X_to_plot, H_to_plot, colors[0], label='initial theta')

			for j in range(1, theta_plots):
				H_to_plot = [self.thetas[j*theta_step].dot(i) for i in X_modified]
				plt.plot(X_to_plot, H_to_plot, colors[j], label='theta after ' + str(j*theta_step) + ' iterations')

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
			ax.scatter(X1, X2, Y, color='red')

			x1_min = min(X1)
			x1_max = max(X1)
			x2_min = min(X2)
			x2_max = max(X2)

			theta_plot_accuracy = 40

			X1_to_plot = np.linspace(x1_min, x1_max, num=theta_plot_accuracy)
			X2_to_plot = np.linspace(x2_min, x2_max, num=theta_plot_accuracy)

			X1_modified = self.normalize([self.add_features(np.array([i])) for i in X1_to_plot], 0)
			X2_modified = self.normalize([self.add_features(np.array([i])) for i in X2_to_plot], 1)

			X1, X2 = np.meshgrid(X1_to_plot, X2_to_plot)

			H_to_plot = np.zeros((X1.shape))
			for h in range(len(X1_modified)):
				for w in range(len(X2_modified)):
					H_to_plot[h][w] = self.theta.dot(np.concatenate((X1_modified[h], X2_modified[w][1:])))

			ax.contour3D(X1, X2, H_to_plot, 100)
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

manager.run(1000)

print('final cost:', manager.compute_cost())
print('final theta:', manager.theta)

manager.plot_data()
manager.plot_costs()