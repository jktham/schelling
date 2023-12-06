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
		self.price_distribution = None
		self.satisfaction_distribution = None
		self.size = size
		self.iterations = iterations
		self.iteration = 0
		self.strategy_weights = strategy_weights
		self.empty_ratio = empty_ratio
		self.history_satisfaction_expensive = [0]*iterations
		self.history_satisfaction_cheap = [0]*iterations
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
			self.history_satisfaction_expensive[i] = self.get_average_satisfaction_expensive()
			self.history_satisfaction_cheap[i] = self.get_average_satisfaction_cheap()
			self.history_satisfaction[i] = self.get_average_satisfaction()


			self.history_time[i] = round(time.time()-t0, 2)

			if i % 20 == 0:
				print(f'iteration: {i}/{self.iterations}, time: {self.history_time[i]}s')
				# Create a histogram of self.satisfaction_distribution
				# Create a histogram of self.satisfaction_distribution
				plt.figure(figsize=(12, 6))
				plt.subplot(131)  # Image of agents
				plt.imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=-1, vmax=self.agent_types)
				plt.title('Agent Types')

				plt.subplot(132)  # Histogram of satisfaction_distribution
				plt.hist(self.satisfaction_distribution, bins=20, color='skyblue', edgecolor='black')
				plt.xlabel('Satisfaction Level')
				plt.ylabel('Frequency')
				plt.title('Satisfaction Distribution')

				#if self.price_distribution!=None:

					#plt.subplot(133)  # Histogram of price_distribution
					#plt.hist(self.price_distribution, bins=20, color='salmon', edgecolor='black')
					#plt.xlabel('Price Level')
					#plt.ylabel('Frequency')
					#plt.title('Price Distribution')

				# Plot 3: Price Distribution Heatmap
				plt.subplot(133)
				plt.imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='plasma')
				plt.title('Price Distribution')
				plt.colorbar()

				plt.tight_layout()
				plt.show()
				plt.close()


			self.iterate()

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
		low_income_agent = 0
		satisfactions = []
		for x in range(self.size):
			for y in range(self.size):
				if (self.agents[x, y].type != -1):
					sat += self.get_satisfaction(x, y)
					low_income_agent += 1
					satisfactions += [self.get_satisfaction(x, y)]
		self.satisfaction_distribution = satisfactions

		return sat / low_income_agent

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
        
		fig = plt.figure()
		fig.suptitle(f'Schelling Model ({self.iteration} iterations)')

		fig.add_subplot(5, 2, 1)
		plt.imshow(np.vectorize(lambda a: a.type)(self.agents), cmap='cool', vmin=-1, vmax=self.agent_types)
		plt.title(f'Agent Types')

		fig.add_subplot(5, 2, 2)
		plt.imshow(np.vectorize(lambda a: a.wealth)(self.agents), cmap='cool', vmin=0, vmax=max(self.agent_wealths))
		plt.title(f'Agent Wealth')

		fig.add_subplot(5, 2, 3)
		plt.imshow(np.vectorize(lambda c: c.type)(self.cells), cmap='cool', vmin=0, vmax=self.cell_types)
		plt.title(f'Cell Types')

		fig.add_subplot(5, 2, 4)
		plt.imshow(np.vectorize(lambda c: c.price)(self.cells), cmap='cool', vmin=0, vmax=max(self.agent_wealths))
		plt.title(f'Cell Prices')

		fig.add_subplot(5, 2, 5)
		plt.imshow(np.vectorize(lambda c: c.distances[0])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 0')

		fig.add_subplot(5, 2, 6)
		plt.imshow(np.vectorize(lambda c: c.distances[1])(self.cells), cmap='cool', vmin=0, vmax=70)
		plt.title(f'Cell Distances 1')

		fig.add_subplot(5, 2, 7)
		plt.plot(range(0, self.iterations), self.history_satisfaction_expensive)
		plt.title(f'Average satisfaction expensive places')
        
		fig.add_subplot(5, 2, 8)
		plt.plot(range(0, self.iterations), self.history_satisfaction_cheap)
		plt.title(f'Average satisfaction cheap places')

		fig.add_subplot(5, 2, 9)
		plt.plot(range(0, self.iterations), self.history_satisfaction)
		plt.title(f'Average satisfaction')
        
		fig.add_subplot(5, 2, 10)
		plt.plot(range(0, self.iterations), self.history_time)
		plt.title(f'Simulation Time')
        
		plt.savefig("plot.png")
		plt.show()

# example model
model = Model(
	size=25,
	iterations=150,
	strategy_weights={
		"random": 0.2,
		"min_price": 0.6,
		"min_point_dist": 0
	},
	empty_ratio=0.3,
	agent_types=3,
	agent_ratios=[1/3]*3,
	agent_thresholds=[0.4, 0.4, 0.4, 0.4, 0.4],
	agent_wealths=[16, 20, 24, 22, 18],
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
