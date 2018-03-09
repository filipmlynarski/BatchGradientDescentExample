import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class GradientDescentManager:

	def __init__(self, X, Y):

		self.data = ([np.array(X), np.array(Y)])
		self.m = len(X)

		self.alpha = 0.0001
		self.theta = []
		self.steps = 1
		self.thetas = np.array([])
		self.costs = []

	def compute_cost(self):
		cost = 0
		for i in range(self.m):
			cost += (self.data[0][i].dot(self.theta) - self.data[1][i]) ** 2
		return 1 / (2*self.m) * cost

	def get_gradient(self):
		gradients = np.zeros(len(self.theta))
		for i in range(self.m):
			gradients[0] += (self.data[0][i].dot(self.theta) - self.data[1][i])
			gradients[1:] += (self.data[0][i].dot(self.theta) - self.data[1][i]) * self.data[0][i][1:]
		return 1 / self.m * gradients

	def run(self, steps):
		for step in range(steps):
			self.steps += 1

			C = self.compute_cost()
			grad = self.get_gradient()

			self.theta -= self.alpha * grad
			self.costs.append(C)
			self.thetas = np.vstack((self.thetas, self.theta))

	def create_theta(self):
		self.theta = np.random.random((len(self.data[0][0]))) * 2 - 1
		self.thetas = [self.theta]

	def plot_data(self):
		plt.plot([i[1] for i in self.data[0]], self.data[1], 'bo')
		for j in range(10):
			plt.plot([i for i in range(30)], [self.thetas[100*j][0] + self.thetas[100*j][1]*i for i in range(30)], 'r')
		plt.plot([i for i in range(30)], [self.thetas[-1][0] + self.thetas[-1][1]*i for i in range(30)], 'g')
		plt.show()

	def plot_costs(self):
		plt.plot([i for i in range(len(self.costs))], self.costs)
		plt.show()

file = open('ex1data1.txt')
content = file.read().splitlines()
file.close()

X = [[1, float(i.split(',')[0])] for i in content]
Y = [float(i.split(',')[1]) for i in content]

manager = GradientDescentManager(X, Y)
manager.create_theta()
manager.theta = np.array([-1.,2.])
print(manager.compute_cost())
manager.run(1000)
print(manager.theta)
print(manager.compute_cost())
manager.theta = np.array([-3.63,1.16])
print(manager.compute_cost())
manager.plot_costs()
manager.plot_data()