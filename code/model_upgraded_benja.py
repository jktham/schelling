import random
import matplotlib.pyplot as plt
import math

class agent:
    """
    Agent class with a similarity threshold and wealth
    
    """
    def __init__(self, i, j, agent_type = None, similarity_threshold = None , wealth = None, price = 0):
        """ Create an agent"""
        self.i = i
        self.j = j
        self.agent_type = random.randint(1, num_agent_types) if agent_type is None else agent_type
        self.similarity_threshold = random.random() if similarity_threshold is None else similarity_threshold
        self.wealth = random.randint(1, 8)  if wealth is None else wealth
        self.price = price
        
        self.similar_neighbours = -1
        self.total_neighbours   = -1
        
        for neighbor_i in range(max(0, self.i-1), min(grid_size-1, self.i+2)):
            for neighbor_j in range(max(0, self.j-1), min(grid_size-1, self.j+2)):
                if grid[neighbor_i][neighbor_j] is not None:
                    if grid[neighbor_i][neighbor_j].agent_type == self.agent_type:
                        self.similar_neighbours += 1
                        self.total_neighbours   += 1
                    else:
                        self.total_neighbours += 1
                    
        self.price = self.total_neighbours
        
        
          
    def move(self):
        """ Calculate if agent likes the position in terms of price and similarity. If not, they move """
        # Generate a list of neighboring cell coordinates around (i, j)
        self.similar_neighbours = -1
        self.total_neighbours   = -1

        for neighbor_i in range(max(0, self.i-1), min(grid_size-1, self.i+2)):
            for neighbor_j in range(max(0, self.j-1), min(grid_size-1, self.j+2)):
                if grid[neighbor_i][neighbor_j] is not None:
                    if grid[neighbor_i][neighbor_j].agent_type == self.agent_type:
                        self.similar_neighbours += 1
                        self.total_neighbours   += 1
                    else:
                        self.total_neighbours += 1
                    
        self.price = self.total_neighbours

        if self.total_neighbours != 0:
            if (math.floor(100*(self.similar_neighbours/self.total_neighbours)) < self.similarity_threshold) and (math.floor(grid[self.i][self.j].price) != self.wealth):
                grid[self.i][self.j] = None
                
                self.i = random.randint(1, grid_size -1)
                self.j = random.randint(1, grid_size -1)
                
                while grid[self.i][self.j] is not None:
                    self.i = random.randint(1, grid_size -1)
                    self.j = random.randint(1, grid_size -1)

                grid[self.i][self.j] = self
        
        if self.total_neighbours <= 0:
            grid[self.i][self.j] = None
            self.i = random.randint(1, grid_size -1)
            self.j = random.randint(1, grid_size -1)   
            while grid[self.i][self.j] is not None:
                self.i = random.randint(1, grid_size -1)
                self.j = random.randint(1, grid_size -1)
            grid[self.i][self.j] = self
            
        if self.similar_neighbours == 0:
            grid[self.i][self.j] = None
                
            self.i = random.randint(1, grid_size -1)
            self.j = random.randint(1, grid_size -1)
                
            while grid[self.i][self.j] is not None:
                self.i = random.randint(1, grid_size -1)
                self.j = random.randint(1, grid_size -1)

            grid[self.i][self.j] = self

# Example usage:
grid_size       = 10
n_agents        = 70
num_agent_types = 2
max_iterations  = 100

if n_agents > (grid_size - 1)**2:
    raise ValueError(f"Too many agents: The number of agents {n_agents} should be lower than {(grid_size-1)**2}.")
# Generate the initial grid with agents and empty cells
grid = [[None for j in range(grid_size)] for k in range(grid_size)]

# Adding agents to the grid:
for i in range(n_agents):
    x = random.randint(1, grid_size - 1)
    y = random.randint(1, grid_size - 1)
    while grid[x][y] is not None:
        x = random.randint(1, grid_size - 1)
        y = random.randint(1, grid_size - 1)
    grid[x][y] = agent(i = x,  j = y, similarity_threshold = 0.5)
    print("Agent installed " + str(i + 1))
    
color_grid = [[0 for j in range(grid_size + 1)] for k in range(grid_size + 1)]
for i in range(grid_size):
    for j in range(grid_size):
        if type(grid[i][j]) is not type(None):
            color_grid[i][j] = grid[i][j].agent_type
plt.imshow(color_grid, cmap='cool', vmin=0, vmax=num_agent_types)
plt.colorbar(ticks=list(range(num_agent_types + 1)), label='Agent Type')

plt.xticks(range(grid_size + 1))
plt.yticks(range(grid_size+ 1))
plt.grid(color='black', linewidth=2)
plt.show()
plt.close()


for iterations in range(max_iterations):
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] is not None:
                grid[i][j].move()
        
    color_grid = [[0 for j in range(grid_size + 1)] for k in range(grid_size + 1)]
    for i in range(grid_size):
        for j in range(grid_size):
            if type(grid[i][j]) is not type(None):
                color_grid[i][j] = grid[i][j].agent_type
    plt.imshow(color_grid, cmap='cool', vmin=0, vmax=num_agent_types)
    plt.colorbar(ticks=list(range(num_agent_types + 1)), label='Agent Type')
                    
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size+ 1))
    plt.grid(color='black', linewidth=2)
    plt.show()
    plt.close()
