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
		self.price = 0
		self.distances = distances

class Point:
	def __init__(self, type, x, y):
		self.type = type
		self.x = x
		self.y = y

class Model:
	def __init__(self, size, iterations, strategy_weights, empty_ratio, agent_types, agent_ratios, agent_thresholds, agent_wealths, cell_types, cell_ratios, point_types, agent_interests):
		self.size = size
		self.iterations = iterations
		self.iteration = 0
		self.strategy_weights = strategy_weights
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

		self.points = np.empty(shape=(point_types), dtype=object)
		self.point_types = point_types

	# fill agent grid according to types and ratios
	def generate_agents(self):
		self.agents.fill(None)
		for x in range(self.size):
			for y in range(self.size):
				if random.random() < self.empty_ratio:
					self.agents[x, y] = Agent(
						type=-1,
						threshold=0,
						wealth=0,
						interests=[0]*self.point_types
					)
				else:
					t = np.random.choice(range(self.agent_types), p=self.agent_ratios)
					self.agents[x, y] = Agent(
						type=t,
						threshold=self.agent_thresholds[t],
						wealth=self.agent_wealths[t] if t != 4 else 2,
						interests=self.agent_interests[t]
					)

	# fill cell grid according to types and ratios
	def generate_cells(self):
		self.cells.fill(None)
		for x in range(self.size):
			for y in range(self.size):
				t = np.random.choice(range(self.cell_types), p=self.cell_ratios)
				self.cells[x, y] = Cell(
					type=t,
					price=0,
					distances=np.empty(shape=(self.point_types))
				)


	# add one point of interest for each type at random locations and update cell distances
	def generate_points(self):
		self.points.fill(None)
		for i in range(len(self.points)):
			self.points[i] = Point(
				type=i,
				x=random.randint(0, self.size-1),
				y=random.randint(0, self.size-1)
			)
		self.update_distances()
        

	# calculate distances to each point type for all cells
	def update_distances(self):
		distances = np.empty((self.size, self.size, len(self.points)))
		for i, point in enumerate(self.points):
			x_diff = np.arange(self.size)[:, None] - point.x
			y_diff = np.arange(self.size) - point.y
			distances[:, :, i] = np.sqrt(x_diff ** 2 + y_diff ** 2)

		for x in range(self.size):
			for y in range(self.size):
				self.cells[x, y].distances = distances[x, y]

	# initialize model, populate agent grid, cell grid, and add points of interest
	def setup(self):
		self.generate_agents()
		self.generate_cells()
		self.generate_points()

	# get neighbors as list of coordinate tuples around (x, y)
	def get_neighbors(self, x, y):
		return [(i, j) for i in range(max(0, x-1), min(x+2, self.size)) for j in range(max(0, y-1), min(y+2, self.size)) if (i, j) != (x, y)]

	# calculate satisfaction for agent at (x, y). currently only cares about type
	def get_satisfaction(self, x, y):
		neighbors = self.get_neighbors(x, y)

		empty_neighbors = sum([(self.agents[neighbor].type == -1) for neighbor in neighbors])
		similar_neighbors = sum([(self.agents[neighbor].type == self.agents[x, y].type) for neighbor in neighbors])

		if empty_neighbors == len(neighbors):
			return 0.0
		return similar_neighbors / (len(neighbors) - empty_neighbors)
    
    # Change price of cells
	def update_price(self):
		for x in range(self.size):
			for y in range(self.size):
				neighbors = self.get_neighbors(x, y)
				self.cells[x,y].price = 8 - sum([(self.agents[neighbor].type == -1) for neighbor in neighbors])


	# check if agent at (x, y) is satisfied or not
	def is_unsatisfied(self, x, y):
		if self.agents[x, y].type == -1:
			return False
		satisfaction = self.get_satisfaction(x, y)
		return satisfaction < self.agents[x, y].threshold
	
	# get desirability score of cell for given agent, depending on strategy weights
	def get_desirability(self, cell, agent):
		desirability = 0
		if self.strategy_weights["random"] > 0.0:
			desirability += random.random() * self.strategy_weights["random"]
		
		if self.strategy_weights["min_price"] > 0.0:
			#desirability += 1 / (cell.price + 0.01) * self.strategy_weights["min_price"]
			desirability += self.strategy_weights["min_price"] if agent.wealth == cell.price else 0
            
		if self.strategy_weights["min_point_dist"] > 0.0:
			weighted_dist_sum = sum([cell.distances[self.points[i].type] * agent.interests[self.points[i].type] for i in range(len(self.points))])
			desirability += 1 / (weighted_dist_sum + 0.01) * self.strategy_weights["min_point_dist"]
			desirability = desirability * agent.wealth == cell.price

		return desirability

	# get empty locations and pick a random most desirable location for agent at (x, y)
	def get_empty_location(self, x, y):
		empty_locations = [(i, j) for i in range(self.size) for j in range(self.size) if (self.agents[i, j].type == -1)]
		if empty_locations:
			desirabilities = [self.get_desirability(self.cells[location], self.agents[x, y]) for location in empty_locations]
			max_desirability = max(desirabilities)
			max_desirable_locations = [empty_locations[index] for index, value in enumerate(desirabilities) if value == max_desirability]
			return random.choice(max_desirable_locations)
			
		return None

	# move agent at (x, y) to new empty location
	def move_agent(self, x, y):
		empty_location = self.get_empty_location(x, y)
		if empty_location:
			swap = self.agents[empty_location]
			self.agents[empty_location] = self.agents[x, y]
			self.agents[x, y] = swap


	# iterate model
	def iterate(self):
		self.iteration += 1
		self.update_price()

		for x in range(self.size):
			for y in range(self.size):
				if self.is_unsatisfied(x, y):
					self.move_agent(x, y)


	# run model for configured number of iterations
	def run(self):
		t0 = time.time()
		for i in range(self.iterations):
			self.history_satisfaction[i] = self.get_average_satisfaction()
			self.history_time[i] = round(time.time()-t0, 2)

			if i % 10 == 0:
				print(f'iteration: {i}/{self.iterations}, time: {self.history_time[i]}s')
				plt.imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=-1, vmax=self.agent_types)
				plt.show()
				plt.close()

			self.iterate()

	# get average satisfaction of all agents
	def get_average_satisfaction(self):
		sat = 0
		for x in range(self.size):
			for y in range(self.size):
				if self.agents[x, y].type != -1:
					sat += self.get_satisfaction(x, y)
		return sat / self.size**2
	
	# display current state of model using matplotlib
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
		plt.imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='cool', vmin=0, vmax=max(self.agent_wealths))
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
		plt.savefig("plot.png")
		plt.show()

# example model
model = Model(
	size=50,
	iterations=300,
	strategy_weights={
		"random": 0,
		"min_price": 0,
		"min_point_dist": 1
	},
	empty_ratio=0.4,
	agent_types=4,
	agent_ratios=[0.25, 0.25, 0.25, 0.25],
	agent_thresholds=[0.7, 0.7, 0.7, 0.7, 0.7],
	agent_wealths=[4, 5, 6, 7, 8],
	cell_types=4,
	cell_ratios=[0.4, 0.3, 0.2, 0.1],
	point_types=2,
	agent_interests=[
		[1.0, 0.0],
		[1.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[0.0, 1.0]
	]
)

model.setup()
model.run()
model.display()
