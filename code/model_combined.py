import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import time
import copy

class Agent:
	"""
        Represents an agent in the simulation.

        Parameters:
        - type (int): The type or category of the agent.
        - threshold (float): The satisfaction threshold of the agent.
        - wealth (float): The wealth level of the agent.
        - interests (list): The agent's interest for points of interest in the grid.
	"""
	def __init__(self, type, threshold, wealth, interests):
		self.type = type
		self.threshold = threshold
		self.wealth = wealth
		self.interests = interests

class Cell:
	"""
    Represents a cell in the environment.

	Parameters:
	- type (int): The type or category of the cell.
	- price (float): The price associated with the cell.
	- distances (numpy.ndarray): An array storing distances to each point type for this cell.
	"""
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
	def __init__(self, size, iterations, strategy_weights, satisfaction_weights, empty_ratio, agent_types, agent_ratios, agent_thresholds, agent_wealths, cell_types, cell_ratios, point_types, points, agent_interests):
		self.size = size
		self.iterations = iterations
		self.iteration = 0
		self.strategy_weights = strategy_weights
		self.satisfaction_weights = satisfaction_weights
		self.empty_ratio = empty_ratio
		self.price_distribution = None
		self.satisfaction_distribution = None
		self.moved = np.zeros(shape=(size, size))
		self.similarity_distribution = None

		self.history_satisfaction_expensive = [0]*iterations
		self.history_satisfaction_cheap = [0]*iterations
		self.history_satisfaction = [0]*iterations
		self.history_time = [0]*iterations
		self.history_agents = [np.empty(shape=(size, size), dtype=object)]*iterations
		self.history_cells = [np.empty(shape=(size, size), dtype=object)]*iterations
		self.history_price_distribution = [None]*iterations
		self.history_satisfaction_distribution = [None]*iterations
		self.history_moved = [np.zeros(shape=(size, size))]*iterations
		self.history_moved_count = [0]*iterations
		self.history_satisfaction_grid = [np.zeros(shape=(size, size))]*iterations
		self.history_similarity = [0]*iterations
		self.history_satisfied_ratio = [0]*iterations

		self.agents = np.empty(shape=(size, size), dtype=object)
		self.agent_types = agent_types
		self.agent_ratios = agent_ratios
		self.agent_thresholds = agent_thresholds
		self.agent_wealths = agent_wealths
		self.agent_interests = agent_interests

		self.cells = np.empty(shape=(size, size), dtype=object)
		self.cell_types = cell_types
		self.cell_ratios = cell_ratios

		self.points = points
		self.point_types = point_types

	# fill agent grid according to types and ratios
	def generate_agents(self):
		"""
		Fill the agent grid with randomly generated agents based on specified parameters.

		The method populates the grid with agents, considering the empty ratio and randomly assigning agent types.

		- For empty locations (determined by the empty ratio), create empty agents.
		- For non-empty locations, randomly assign agent types based on given ratios.

		"""
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
						wealth=self.agent_wealths[t] if t != 3 else 8,
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
		for p in self.points:
			if p.x == None:
				p.x = random.randint(0, self.size-1)
			if p.y == None:
				p.y = random.randint(0, self.size-1)
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

	# calculate similarity for agent at (x, y). only cares about type
	def get_similarity(self, x, y):
		neighbors = self.get_neighbors(x, y)
		empty_neighbors = sum([(self.agents[neighbor].type == -1) for neighbor in neighbors])
		similar_neighbors = sum([(self.agents[neighbor].type == self.agents[x, y].type) for neighbor in neighbors])

		if empty_neighbors == len(neighbors):
			return 0.0
		return similar_neighbors / (len(neighbors) - empty_neighbors)
	
	# calculate satisfaction for agent at (x, y)
	def get_satisfaction(self, x, y):
		similarity = self.get_similarity(x, y)
		desirability = self.get_desirability(self.cells[x, y], self.agents[x, y])
		return self.satisfaction_weights[0] * similarity + self.satisfaction_weights[1] * desirability
	
	# get neighbors as list of coordinate tuples around (x, y)
	def get_price(self, x, y):
		return [(i, j) for i in range(max(0, x-2), min(x+3, self.size)) for j in range(max(0, y-2), min(y+3, self.size)) if (i, j) != (x, y)]

	# Change price of cells
	def update_price(self):
		prices = []
		for x in range(self.size):
			for y in range(self.size):
				neighbors = self.get_neighbors(x, y)
				self.cells[x,y].price = 24 - sum([(self.agents[neighbor].type == -1) for neighbor in neighbors])
				prices += [self.cells[x,y].price]

		print(np.median(prices), np.min(prices), np.max(prices), np.mean(prices), np.std(prices))
		self.price_distribution = prices

	# check if agent at (x, y) is satisfied or not
	def is_unsatisfied(self, x, y):
		if self.agents[x, y].type == -1:
			return False
		satisfaction = self.get_satisfaction(x, y)
		if self.strategy_weights["min_price"] > 0.0:
			return (satisfaction < self.agents[x, y].threshold) and (self.agents[x, y].wealth <= self.cells[x,y].price)
		else:
			return (satisfaction < self.agents[x, y].threshold)         
	
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
			desirability += 1 / (weighted_dist_sum + 1.0) * self.strategy_weights["min_point_dist"]
			# desirability = desirability * agent.wealth == cell.price

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
			self.moved[empty_location] = 1

	# iterate model
	def iterate(self):
		self.moved.fill(0)
		self.iteration += 1
		self.update_price()
		for x in range(self.size):
			for y in range(self.size):
				if self.is_unsatisfied(x, y) and self.moved[x, y] == 0:
					self.move_agent(x, y)

	# run model for configured number of iterations
	def run(self):
		t0 = time.time()
		self.update_price()
		for i in range(self.iterations):
			self.history_satisfaction_expensive[i] = self.get_average_satisfaction_expensive()
			self.history_satisfaction_cheap[i] = self.get_average_satisfaction_cheap()
			self.history_satisfaction[i] = self.get_average_satisfaction()
			self.history_time[i] = round(time.time()-t0, 2)
			self.history_agents[i] = copy.deepcopy(self.agents) # deepcopy might be too slow, benchmark later
			self.history_cells[i] = copy.deepcopy(self.cells)
			self.history_price_distribution[i] = np.copy(self.price_distribution)
			self.history_satisfaction_distribution[i] = np.copy(self.satisfaction_distribution)
			self.history_moved[i] = np.copy(self.moved)
			self.history_moved_count[i] = np.sum(self.moved)
			self.history_satisfaction_grid[i] = self.get_satisfaction_grid()
			self.history_similarity[i] = self.get_average_similarity()
			self.history_satisfied_ratio[i] = self.get_satisfied_ratio()
			
			if i % 10 == 0:
				print(f'iteration: {i}/{self.iterations}, time: {self.history_time[i]}s')

			self.iterate()

	def get_satisfaction_grid(self):
		satisfaction_grid = np.zeros(shape=(self.size, self.size))
		for x in range(self.size):
			for y in range(self.size):
				satisfaction_grid[x, y] = self.get_satisfaction(x, y)
		return satisfaction_grid

	# get average satisfaction of all agents
	def get_average_satisfaction_expensive(self):
		sat = 0
		high_income_agent = 1
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1) and (self.cells[x,y].price > 19):
					sat += self.get_satisfaction(x, y)
					high_income_agent += 1
		return sat / high_income_agent
	
	# get average satisfaction of all agents
	def get_average_satisfaction_cheap(self):
		sat = 0
		low_income_agent = 1
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1) and (self.cells[x,y].price < 19):
					sat += self.get_satisfaction(x, y)
					low_income_agent += 1
		return sat / low_income_agent
	
	# get average satisfaction of all agents
	def get_average_satisfaction(self):
		sat = 0
		agent = 0
		satisfactions = []
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1):
					sat += self.get_satisfaction(x, y)
					agent += 1
					satisfactions += [self.get_satisfaction(x, y)]
		self.satisfaction_distribution = satisfactions
		return sat / agent

	# get average similarity
	def get_average_similarity(self):
		sim = 0
		agent = 0
		similarities = []
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1):
					sim += self.get_similarity(x, y)
					agent += 1
					similarities += [self.get_similarity(x, y)]
		self.similarity_distribution = similarities
		return sim / agent
	
	# get ratio of satisfied agents
	def get_satisfied_ratio(self):
		sat_agents = 0
		agents = 0
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1):
					if (not self.is_unsatisfied(x, y)):
						sat_agents += 1
					agents += 1
		return sat_agents / agents
	
	def get_satisfaction_distribution(self):
		sat = []
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1):
					sat.append(self.get_satisfaction(x, y))
		return sat

	def get_price_distribution(self):
		price = []
		for x in range(self.size):
			for y in range(self.size):
				price.append(self.cells[x,y].price)
		return price
	
	# display current state of model using matplotlib
	def display(self):
		fig, ax = plt.subplots(2, 5, figsize=(16, 8))
		fig.tight_layout(rect=[0, 0.02, 1, 0.92])
		fig.suptitle(f'size: {self.size}, iterations: {self.iterations}, strategy: ({self.strategy_weights["random"]}, {self.strategy_weights["min_price"]}, {self.strategy_weights["min_point_dist"]}), satisfaction: {self.satisfaction_weights}, empty: {self.empty_ratio}, \ntypes: {self.agent_types}, ratios: {np.round(self.agent_ratios, 2)}, thresholds: {self.agent_thresholds}, wealths: {self.agent_wealths}, interests: {self.agent_interests}', wrap=True)

		ax[0, 0].set_title(f'Final agents')
		ax[0, 0].imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=-1, vmax=self.agent_types)

		ax[0, 1].set_title(f'Final prices')
		ax[0, 1].imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='plasma', vmin=16, vmax=24)

		ax[0, 2].set_title(f'Moves over time')
		ax[0, 2].plot(range(0, self.iterations), self.history_moved_count)
		ax[0, 2].set_box_aspect(1)

		ax[0, 3].set_title(f'Satisfaction over time')
		ax[0, 3].plot(range(0, self.iterations), self.history_satisfaction, label="avg. satisfaction")
		ax[0, 3].plot(range(0, self.iterations), self.history_satisfaction_cheap, label="avg. sat cheap")
		ax[0, 3].plot(range(0, self.iterations), self.history_satisfaction_expensive, label="avg. sat expensive")
		ax[0, 3].plot(range(0, self.iterations), self.history_similarity, label="avg. similarity")
		ax[0, 3].plot(range(0, self.iterations), self.history_satisfied_ratio, label="satisfied ratio")
		ax[0, 3].legend()
		ax[0, 3].set_box_aspect(1)

		ax[0, 4].set_title(f'Final satisfaction distr.')
		ax[0, 4].hist(self.satisfaction_distribution, bins=20, color='skyblue', edgecolor='black')
		ax[0, 4].set_xlim([0, 1])
		ax[0, 4].set_box_aspect(1)

		ax[1, 0].set_title(f'Agents at iteration 1/{self.iterations}')
		ax[1, 0].imshow(np.vectorize(lambda a: a.type)(self.history_agents[0]), cmap='cool', vmin=-1, vmax=self.agent_types)

		ax[1, 1].set_title(f'Prices at iteration 1/{self.iterations}')
		ax[1, 1].imshow(np.vectorize(lambda c: c.price)(self.history_cells[0]), cmap='plasma', vmin=16, vmax=24)

		ax[1, 2].set_title(f'Moves at iteration 1/{self.iterations}')
		ax[1, 2].imshow(self.history_moved[0], cmap='hot', vmin=0, vmax=1)

		ax[1, 3].set_title(f'Satisfaction at iteration 1/{self.iterations}')
		ax[1, 3].imshow(self.history_satisfaction_grid[0], cmap='plasma', vmin=0, vmax=1)

		ax[1, 4].set_title(f'Points of interest')
		for p in self.points:
			ax[1, 4].scatter(p.y, p.x, label=p.type)
		ax[1, 4].legend()
		ax[1, 4].set_xlim([0, self.size-1])
		ax[1, 4].set_ylim([0, self.size-1][::-1])
		ax[1, 4].set_box_aspect(1)

		def animate(frame):
			ax[1, 0].set_title(f'Agents at iteration {frame+1}/{self.iterations}')
			ax[1, 0].get_images()[0].set_data(np.vectorize(lambda a: a.type)(self.history_agents[frame]))

			ax[1, 1].set_title(f'Prices at iteration {frame+1}/{self.iterations}')
			ax[1, 1].get_images()[0].set_data(np.vectorize(lambda c: c.price)(self.history_cells[frame]))

			ax[1, 2].set_title(f'Moves at iteration {frame+1}/{self.iterations}')
			ax[1, 2].get_images()[0].set_data(self.history_moved[frame])

			ax[1, 3].set_title(f'Satisfaction at iteration {frame+1}/{self.iterations}')
			ax[1, 3].get_images()[0].set_data(self.history_satisfaction_grid[frame])

		anim = animation.FuncAnimation(fig=fig, func=animate, frames=self.iterations, interval=50)

		i = 1
		while os.path.exists(f'plots/plot_{str(i).zfill(3)}.png') or os.path.exists(f'plots/plot_{str(i).zfill(3)}.gif'):
			i += 1

		# fig.savefig(f'plots/plot_{str(i).zfill(3)}.png', dpi=144)
		anim.save(f'plots/plot_{str(i).zfill(3)}.gif', dpi=144, fps=20, writer="pillow")

		# plt.show()

# example model
model = Model(
	size=100,
	iterations=100,
	strategy_weights={
		"random": 0.4,
		"min_price": 0.2,
		"min_point_dist": 0.4
	},
	satisfaction_weights=[
		0.5, # similarity
		0.5 # desirability
	],
	empty_ratio=0.2,
	agent_types=5,
	agent_ratios=[1/5]*5,
	agent_thresholds=[0.3]*5,
	agent_wealths=[16, 18, 19, 22, 24],
	cell_types=4,
	cell_ratios=[0.4, 0.3, 0.2, 0.1],
	point_types=4,
	points=[
		Point(type=0, x=30, y=30),
		Point(type=1, x=30, y=70),
		Point(type=2, x=75, y=50),
		Point(type=3, x=50, y=50)
	],
	agent_interests=[
		[0.0, 0.0, 0.0, 1.0],
		[0.3, 0.0, 0.0, 0.8],
		[0.0, 0.3, 0.0, 0.8],
		[0.0, 0.0, 0.3, 0.8],
		[0.0, 0.0, 0.0, 1.0]
	]
)

model.setup()
model.run()
model.display()
