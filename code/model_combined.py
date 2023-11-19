import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class Agent:
	def __init__(self, type, threshold, wealth):
		self.type = type
		self.threshold = threshold
		self.wealth = wealth

class Cell:
	def __init__(self, type, price, point_distances):
		self.type = type
		self.price = price
		self.point_distances = point_distances

class Point:
	def __init__(self, type, i, j):
		self.type = type
		self.i = i
		self.j = j

class Model:
	def __init__(self, size, iterations, strategy, agent_types, agent_ratios, agent_thresholds, agent_wealths, cell_types, cell_ratios, cell_prices, point_count):
		self.size = size
		self.iterations = iterations
		self.iteration = 0
		self.strategy = strategy

		self.agents = np.empty(shape=(size, size), dtype=object)
		self.agent_types = agent_types
		self.agent_ratios = agent_ratios
		self.agent_thresholds = agent_thresholds
		self.agent_wealths = agent_wealths

		self.cells = np.empty(shape=(size, size), dtype=object)
		self.cell_types = cell_types
		self.cell_ratios = cell_ratios
		self.cell_prices = cell_prices

		self.points = np.empty(shape=(point_count), dtype=object)
		self.point_count = point_count

	def setup(self):
		self.agents.fill(None)
		for i in range(self.size):
			for j in range(self.size):
				t = np.random.choice(range(self.agent_types+1), p=self.agent_ratios)
				self.agents[i][j] = Agent(
					type=t, 
					threshold=self.agent_thresholds[t], 
					wealth=self.agent_wealths[t]
				)
		
		self.cells.fill(None)
		for i in range(self.size):
			for j in range(self.size):
				t = np.random.choice(range(self.cell_types), p=self.cell_ratios)
				self.cells[i][j] = Cell(
					type=t, 
					price=self.cell_prices[t], 
					point_distances=np.empty(shape=(self.point_count))
				)

		self.points.fill(None)
		for p in range(self.point_count):
			self.points[p] = Point(
				type=0, 
				i=random.randint(0, self.size-1), 
				j=random.randint(0, self.size-1)
			)
			for i in range(self.size):
				for j in range(self.size):
					self.cells[i][j].point_distances[p] = math.sqrt(abs(i - self.points[p].i)**2 + abs(j - self.points[p].j)**2)

	def get_neighbors(self, i, j):
		return [(k, q) for k in range(max(0, i-1), min(i+2, self.size)) for q in range(max(0, j-1), min(j+2, self.size)) if (k, q) != (i, j)]

	def get_satisfaction(self, i, j):
		neighbors = self.get_neighbors(i, j)

		empty_neighbors = sum([(self.agents[neighbor].type == 0) for neighbor in neighbors])
		similar_neighbors = sum([(self.agents[neighbor].type == self.agents[i, j].type) for neighbor in neighbors])

		if empty_neighbors == len(neighbors):
			return 0.0
		return similar_neighbors / (len(neighbors) - empty_neighbors)

	def is_unsatisfied(self, i, j):
		if self.agents[i, j].type == 0:
			return False
		satisfaction = self.get_satisfaction(i, j)
		return satisfaction < self.agents[i, j].threshold
	
	def get_empty_location(self, i, j):
		empty_cells = [(x, y) for x in range(self.size) for y in range(self.size) if (self.agents[x, y].type == 0)]
		if empty_cells:
			if self.strategy == "random":
				return random.choice(empty_cells)
			if self.strategy == "min_price":
				empty_cells_prices = {(i, j): self.cells[i, j].price for i, j in empty_cells}
				min_price = min(empty_cells_prices.values())
				min_price_cells = [key for key, value in empty_cells_prices.items() if value == min_price]
				return random.choice(min_price_cells)
			if self.strategy == "min_dist_0":
				empty_cells_prices = {(i, j): self.cells[i, j].point_distances[0] for i, j in empty_cells}
				min_price = min(empty_cells_prices.values())
				min_price_cells = [key for key, value in empty_cells_prices.items() if value == min_price]
				return random.choice(min_price_cells)
		return None

	def move_agent(self, i, j):
		empty_location = self.get_empty_location(i, j)
		if empty_location:
			swap = self.agents[empty_location]
			self.agents[empty_location] = self.agents[i, j]
			self.agents[i, j] = swap

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
		fig = plt.figure()
		fig.suptitle(f'Schelling Model ({self.iteration} iterations)')

		fig.add_subplot(3, 2, 1)
		plt.imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=0, vmax=self.agent_types)
		plt.title(f'Agent Types')

		fig.add_subplot(3, 2, 2)
		plt.imshow(np.vectorize(lambda a: a.wealth)(self.agents), cmap='cool', vmin=0, vmax=max(self.agent_wealths))
		plt.title(f'Agent Wealth')

		fig.add_subplot(3, 2, 3)
		plt.imshow(np.vectorize(lambda c: c.type)(self.cells), cmap='cool', vmin=0, vmax=self.cell_types)
		plt.title(f'Cell Types')

		fig.add_subplot(3, 2, 4)
		plt.imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='cool', vmin=0, vmax=max(self.cell_prices))
		plt.title(f'Cell Prices')

		fig.add_subplot(3, 2, 5)
		plt.imshow(np.vectorize(lambda c: c.point_distances[0])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 0')

		fig.add_subplot(3, 2, 6)
		plt.imshow(np.vectorize(lambda c: c.point_distances[1])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 1')

		plt.show()


model = Model(
	size=50, 
	iterations=200, 
	strategy="min_dist_0", 
	agent_types=5, 
	agent_ratios=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16], 
	agent_thresholds=[0, 0.5, 0.5, 0.5, 0.5, 0.5], 
	agent_wealths=[0, 100, 200, 300, 400, 500], 
	cell_types=4, 
	cell_ratios=[0.4, 0.3, 0.2, 0.1], 
	cell_prices=[100, 200, 300, 400], 
	point_count=2
)

model.setup()
t0 = time.time()
model.run()
t1 = time.time()
print(f'time: {round(t1-t0, 2)}s')
model.display()
