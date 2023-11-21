import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class Agent:
	def __init__(self, type, threshold, wealth, interests):
		self.type = type
		self.threshold = threshold
		self.wealth = wealth
		self.interests = interests

class Cell:
	def __init__(self, type, price, distances):
		self.type = type
		self.price = price
		self.distances = distances

class Point:
	def __init__(self, type, i, j):
		self.type = type
		self.i = i
		self.j = j

class Model:
	def __init__(self, size, iterations, strategy, empty_ratio, agent_types, agent_ratios, agent_thresholds, agent_wealths, cell_types, cell_ratios, cell_prices, point_types, agent_interests):
		self.size = size
		self.iterations = iterations
		self.iteration = 0
		self.strategy = strategy
		self.empty_ratio = empty_ratio
		self.history_satisfaction = [0]*iterations
		self.history_time = [0]*iterations

		self.agents = np.empty(shape=(size, size), dtype=object)
		self.agent_types = agent_types
		self.agent_ratios = agent_ratios
		self.agent_thresholds = agent_thresholds
		self.agent_wealths = agent_wealths
		self.agent_interests = agent_interests

		self.cells = np.empty(shape=(size, size), dtype=object)
		self.cell_types = cell_types
		self.cell_ratios = cell_ratios
		self.cell_prices = cell_prices

		self.points = np.empty(shape=(point_types), dtype=object)
		self.point_types = point_types

	def generate_agents(self):
		self.agents.fill(None)
		for i in range(self.size):
			for j in range(self.size):
				if random.random() < self.empty_ratio:
					self.agents[i, j] = Agent(
						type=-1, 
						threshold=0, 
						wealth=0, 
						interests=[0]*self.point_types
					)
				else:
					t = np.random.choice(range(self.agent_types), p=self.agent_ratios)
					self.agents[i, j] = Agent(
						type=t, 
						threshold=self.agent_thresholds[t], 
						wealth=self.agent_wealths[t], 
						interests=self.agent_interests[t]
					)

	def generate_cells(self):
		self.cells.fill(None)
		for i in range(self.size):
			for j in range(self.size):
				t = np.random.choice(range(self.cell_types), p=self.cell_ratios)
				self.cells[i, j] = Cell(
					type=t, 
					price=self.cell_prices[t], 
					distances=np.empty(shape=(self.point_types))
				)

	def generate_points(self):
		self.points.fill(None)
		for p in range(self.point_types):
			self.points[p] = Point(
				type=0, 
				i=random.randint(0, self.size-1), 
				j=random.randint(0, self.size-1)
			)
		self.update_distances()

	def update_distances(self):
		for p in range(self.point_types):
			for i in range(self.size):
				for j in range(self.size):
					self.cells[i, j].distances[p] = math.sqrt(abs(i - self.points[p].i)**2 + abs(j - self.points[p].j)**2)

	def setup(self):
		self.generate_agents()
		self.generate_cells()
		self.generate_points()

	def get_neighbors(self, i, j):
		return [(k, q) for k in range(max(0, i-1), min(i+2, self.size)) for q in range(max(0, j-1), min(j+2, self.size)) if (k, q) != (i, j)]

	def get_satisfaction(self, i, j):
		neighbors = self.get_neighbors(i, j)

		empty_neighbors = sum([(self.agents[neighbor].type == -1) for neighbor in neighbors])
		similar_neighbors = sum([(self.agents[neighbor].type == self.agents[i, j].type) for neighbor in neighbors])

		if empty_neighbors == len(neighbors):
			return 0.0
		return similar_neighbors / (len(neighbors) - empty_neighbors)

	def is_unsatisfied(self, i, j):
		if self.agents[i, j].type == -1:
			return False
		satisfaction = self.get_satisfaction(i, j)
		return satisfaction < self.agents[i, j].threshold
	
	def get_desirability(self, cell, agent):
		if self.strategy == "random":
			return 1
		
		if self.strategy == "min_price":
			return 1 / (cell.price + 0.01)
		
		if self.strategy == "min_dist":
			weighted_dist_sum = sum([cell.distances[p] * agent.interests[p] for p in range(self.point_types)])
			return 1 / (weighted_dist_sum + 0.01)

	def get_empty_location(self, i, j):
		empty_locations = [(x, y) for x in range(self.size) for y in range(self.size) if (self.agents[x, y].type == -1)]
		if empty_locations:
			desirabilities = [self.get_desirability(self.cells[location], self.agents[i, j]) for location in empty_locations]
			max_desirability = max(desirabilities)
			max_desirable_locations = [empty_locations[index] for index, value in enumerate(desirabilities) if value == max_desirability]
			return random.choice(max_desirable_locations)
			
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

	def get_average_satisfaction(self):
		sat = 0
		for i in range(self.size):
			for j in range(self.size):
				if self.agents[i, j].type != -1:
					sat += self.get_satisfaction(i, j)
		return sat / self.size**2

	def run(self):
		t0 = time.time()
		for i in range(self.iterations):
			self.history_satisfaction[i] = self.get_average_satisfaction()
			self.history_time[i] = round(time.time()-t0, 2)
			if i % 10 == 0: print(f'iteration: {i}/{self.iterations}, time: {self.history_time[i]}s')
			self.iterate()

	def display(self):
		fig = plt.figure()
		fig.suptitle(f'Schelling Model ({self.iteration} iterations)')

		fig.add_subplot(4, 2, 1)
		plt.imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=-1, vmax=self.agent_types)
		plt.title(f'Agent Types')

		fig.add_subplot(4, 2, 2)
		plt.imshow(np.vectorize(lambda a: a.wealth)(self.agents), cmap='cool', vmin=0, vmax=max(self.agent_wealths))
		plt.title(f'Agent Wealth')

		fig.add_subplot(4, 2, 3)
		plt.imshow(np.vectorize(lambda c: c.type)(self.cells), cmap='cool', vmin=0, vmax=self.cell_types)
		plt.title(f'Cell Types')

		fig.add_subplot(4, 2, 4)
		plt.imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='cool', vmin=0, vmax=max(self.cell_prices))
		plt.title(f'Cell Prices')

		fig.add_subplot(4, 2, 5)
		plt.imshow(np.vectorize(lambda c: c.distances[0])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 0')

		fig.add_subplot(4, 2, 6)
		plt.imshow(np.vectorize(lambda c: c.distances[1])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 1')

		fig.add_subplot(4, 2, 7)
		plt.plot(range(0, self.iterations), self.history_satisfaction)
		plt.title(f'Average Satisfaction')

		fig.add_subplot(4, 2, 8)
		plt.plot(range(0, self.iterations), self.history_time)
		plt.title(f'Simulation Time')

		plt.show()


model = Model(
	size=50,
	iterations=300,
	strategy="min_dist",
	empty_ratio=0.2,
	agent_types=5,
	agent_ratios=[0.2, 0.2, 0.2, 0.2, 0.2],
	agent_thresholds=[0.5, 0.5, 0.5, 0.5, 0.5],
	agent_wealths=[100, 200, 300, 400, 500],
	cell_types=4,
	cell_ratios=[0.4, 0.3, 0.2, 0.1],
	cell_prices=[100, 200, 300, 400],
	point_types=2,
	agent_interests=[
		[1, 0],
		[1, 0],
		[1, 0],
		[1, 0],
		[0, 1]
	]
)

model.setup()
model.run()
model.display()
