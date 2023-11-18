import numpy as np
import random
import matplotlib.pyplot as plt

# Define the parameters
grid_size = 20  # Size of the grid (e.g., 20x20)
empty_ratio = 0.2  # Ratio of empty cells
similarity_threshold = 0.5  # Minimum similarity to be happy
num_agent_types = 5

# Calculate the probabilities for agent types
# 0 represents empty cells, and we distribute the rest evenly among the rest of agent types
# if we wanted to add more agents, we only need to change num_agent_types 

agent_prob = (1 - empty_ratio) / num_agent_types  # Probability for each type of agent

probs = [agent_prob]*num_agent_types
# append the empty ratio at the begining of the list 
probs.insert(0, empty_ratio)

# Create the grid with the correct probabilities
grid = np.random.choice(list(range(num_agent_types+1)), size=(grid_size, grid_size), p=probs)

def calculate_similarity(grid, i, j):
    # Calculate the similarity of an agent with its neighbors

    # Generate a list of neighboring cell coordinates around (i, j)
    neighbors = [(k, q) for k in range(max(0, i - 1), min(i + 2, grid_size)) for q in range(max(0, j - 1), min(j + 2, grid_size)) if (k, q) != (i, j)]

    # Count empty neighbors and similar neighbors
    empty_neighbors = sum([grid[neighbor] == 0 for neighbor in neighbors])
    similar_neighbors = sum([grid[neighbor] == grid[i, j] for neighbor in neighbors])

    # Calculate similarity as the ratio of similar neighbors to total neighbors (excluding empty ones)
    
    # if all neighbors are empty, return 0
    if empty_neighbors == len(neighbors):
        return 0.0
    
    # else 
    return similar_neighbors / (len(neighbors) - empty_neighbors)

def is_unsatisfied(grid, i, j):
    # Check if an agent is unsatisfied and wants to move
    if grid[i, j] == 0:
        return False  # Empty cells are always satisfied
    similarity = calculate_similarity(grid, i, j)
    # If the similarity is below the threshold, the agent is unsatisfied
    return similarity < similarity_threshold

def find_empty_location(grid):
    
    # Find a random empty cell
    
    # This is the function that we should modify to add all the constraints when changing cells!
    # then it would not be random choice
    empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i, j] == 0]
    if empty_cells:
        return random.choice(empty_cells)
    return None

def move_agent(grid, i, j):
    # Move an agent to a random empty cell
    empty_location = find_empty_location(grid)
    if empty_location:
        # Move the agent to the empty location and clear the agent's original location
        grid[empty_location] = grid[i, j]
        grid[i, j] = 0

def schelling_model(grid, max_iterations):
    # Run the Schelling model for a specified number of iterations
    for iteration in range(max_iterations):
        for i in range(grid_size):
            for j in range(grid_size):
                if is_unsatisfied(grid, i, j):
                    move_agent(grid, i, j)

# Simulate the Schelling model
max_iterations = 300
schelling_model(grid, max_iterations)
# Visualize the grid
plt.imshow(grid, cmap='cool', vmin=0, vmax=num_agent_types)
plt.colorbar(ticks=list(range(num_agent_types+1)), label='Agent Type')
plt.title(f'Schelling Model ({max_iterations} iterations)')
plt.show()
