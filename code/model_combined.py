import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class Model:
	def __init__(self, size, types, iterations, satisfaction_threshold, empty_ratio):
		self.size = size
		self.grid = np.empty(shape=(size, size))
		self.types = types
		self.iterations = iterations
		self.satisfaction_threshold = satisfaction_threshold
		self.empty_ratio = empty_ratio
		self.iteration = 0

	def setup(self):
		self.grid.fill(0)
		for i in range(self.size):
			for j in range(self.size):
				if random.random() >= self.empty_ratio:
					self.grid[i][j] = random.randint(1, self.types)

	def get_neighbors(self, i, j):
		return [(k, q) for k in range(max(0, i-1), min(i+2, self.size)) for q in range(max(0, j-1), min(j+2, self.size)) if (k, q) != (i, j)]

	def get_satisfaction(self, i, j):
		neighbors = self.get_neighbors(i, j)

		empty_neighbors = sum([(self.grid[neighbor] == 0) for neighbor in neighbors])
		similar_neighbors = sum([(self.grid[neighbor] == self.grid[i, j]) for neighbor in neighbors])

		if empty_neighbors == len(neighbors):
			return 0.0
		return similar_neighbors / (len(neighbors) - empty_neighbors)

	def is_unsatisfied(self, i, j):
		if self.grid[i, j] == 0:
			return False
		satisfaction = self.get_satisfaction(i, j)
		return satisfaction < self.satisfaction_threshold
	
	def get_empty_location(self, i, j):
		empty_cells = [(x, y) for x in range(self.size) for y in range(self.size) if (self.grid[x, y] == 0)]
		if empty_cells:
			return random.choice(empty_cells)
		return None

	def move_agent(self, i, j):
		empty_location = self.get_empty_location(i, j)
		if empty_location:
			self.grid[empty_location] = self.grid[i, j]
			self.grid[i, j] = 0

	def iterate(self):
		self.iteration += 1
		for i in range(self.size):
			for j in range(self.size):
				if self.is_unsatisfied(i, j):
					self.move_agent(i, j)

	def run(self):
		for i in range(self.iterations):
			self.iterate()

	def display(self):
		plt.imshow(self.grid, cmap='cool', vmin=0, vmax=self.types)
		plt.colorbar(ticks=list(range(self.types+1)), label='Agent Type')
		plt.title(f'Schelling Model ({self.iteration} iterations)')
		plt.show()


model = Model(50, 5, 200, 0.5, 0.2)
model.setup()
t0 = time.time()
model.run()
t1 = time.time()
print(f'time: {round(t1-t0, 2)}s')
model.display()
