import numpy as np
import random
import matplotlib.pyplot as plt

def generate_general_grid(grid_size, empty_ratio, num_agent_types):
    """
    Generate a grid with agents and empty cells based on the given parameters.

    Parameters:
    - grid_size (int): Size of the grid (e.g., 20x20).
    - empty_ratio (float): Ratio of empty cells.
    - num_agent_types (int): Number of agent types (excluding empty cells).

    Returns:
    - np.ndarray: The generated grid.
    """
    # Calculate the probabilities for agent types
    # 0 represents empty cells, and we distribute the rest evenly among the rest of agent types
    agent_prob = (1 - empty_ratio) / num_agent_types
    probs = [agent_prob] * num_agent_types
    probs.insert(0, empty_ratio)

    grid = np.random.choice(list(range(num_agent_types + 1)), size=(grid_size, grid_size), p=probs)
    
    return grid

def generate_price_grid(grid_size, prices, probs_prices):
    """
    Generate a grid of prices based on the given parameters.

    Parameters:
    - grid_size (int): Size of the grid (e.g., 20x20).
    - prices (list): List of prices for different agent types.
    - probs_prices (list): Probabilities corresponding to the prices. Must sum to 1.

    Returns:
    - np.ndarray: The generated grid of prices.
    """
    assert np.round(sum(probs_prices), 3) == 1
    grid_prices = np.random.choice(prices, size=(grid_size, grid_size), p=probs_prices)
    
    return grid_prices

def calculate_similarity(grid, i, j):
    """
    Calculate the similarity of an agent with its neighbors.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - i, j (int): Coordinates of the agent.

    Returns:
    - float: Similarity ratio.
    """
    # Generate a list of neighboring cell coordinates around (i, j)
    neighbors = [(k, q) for k in range(max(0, i - 1), min(i + 2, grid_size)) for q in range(max(0, j - 1), min(j + 2, grid_size)) if (k, q) != (i, j)]

    # Count empty neighbors and similar neighbors
    empty_neighbors = sum([grid[neighbor] == 0 for neighbor in neighbors])
    similar_neighbors = sum([grid[neighbor] == grid[i, j] for neighbor in neighbors])

    # Calculate similarity as the ratio of similar neighbors to total neighbors (excluding empty ones)
    if empty_neighbors == len(neighbors):
        return 0.0
    else:
        return similar_neighbors / (len(neighbors) - empty_neighbors)

def calculate_price_satisfaction(price):
    """
    Calculate the satisfaction based on the price.

    Parameters:
    - price (float): The price of the house.

    Returns:
    - float: Price satisfaction.
    """
    return 1 / price

def calculate_total_satisfaction(grid, i, j, agent_weights):
    """
    Calculate the total satisfaction of an agent, considering both similarity and price.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - i, j (int): Coordinates of the agent.
    - agent_weights (dict): Weight parameters for the agent.

    Returns:
    - float: Total satisfaction.
    """
    agent_type = grid[i, j]
    similarity = calculate_similarity(grid, i, j)
    price = grid_prices[i, j]
    price_satisfaction = calculate_price_satisfaction(price)
    
    # Calculate total satisfaction as a weighted sum of price_satisfaction and similarity
    total_satisfaction = agent_weights[agent_type]['weight_price'] * price_satisfaction + agent_weights[agent_type]['weight_similarity'] * similarity
    
    return total_satisfaction

def is_unsatisfied(grid, i, j, agent_thresholds, agent_weights):
    """
    Check if an agent is unsatisfied and wants to move.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - i, j (int): Coordinates of the agent.
    - agent_thresholds (dict): Satisfaction thresholds for each agent type.
    - agent_weights (dict): Weight parameters for the agent.

    Returns:
    - bool: True if the agent is unsatisfied, False otherwise.
    """
    agent_type = grid[i, j]
    satisfaction_threshold = agent_thresholds[agent_type]
    total_satisfaction = calculate_total_satisfaction(grid, i, j, agent_weights)
    
    # If the total satisfaction is below the agent's threshold, the agent is unsatisfied
    return total_satisfaction < satisfaction_threshold

def find_empty_location(grid, strategy):
    """
    Find an empty cell based on the specified strategy.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - strategy (str): The strategy for finding an empty cell.

    Returns:
    - tuple or None: Coordinates of the chosen empty cell or None if no empty cell is found.
    """
    allowed_strategies = ["random", "min_price"]
    if strategy not in allowed_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Allowed strategies are {', '.join(allowed_strategies)}.")

    empty_cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i, j] == 0]
    
    if empty_cells:
        if strategy == "random":
            return random.choice(empty_cells)

        if strategy == "min_price":
            empty_cells_prices = {(i, j): grid_prices[i, j] for i, j in empty_cells}
            min_price = min(empty_cells_prices.values())
            min_price_cells = [key for key, value in empty_cells_prices.items() if value == min_price]
            return random.choice(min_price_cells)

    return None

def move_agent(grid, i, j, strategy):
    """
    Move an agent to a random empty cell based on the specified strategy.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - i, j (int): Coordinates of the agent.
    - strategy (str): The strategy for finding an empty cell.
    """
    empty_location = find_empty_location(grid, strategy)
    if empty_location:
        # Move the agent to the empty location and clear the agent's original location
        grid[empty_location] = grid[i, j]
        grid[i, j] = 0

def schelling_model(grid, max_iterations, strategy, agent_thresholds, agent_weights):
    """
    Run the Schelling model for a specified number of iterations.

    Parameters:
    - grid (np.ndarray): The grid representing the neighborhood.
    - max_iterations (int): The maximum number of iterations to run the model.
    - strategy (str): The strategy for finding an empty cell.
    - agent_thresholds (dict): Satisfaction thresholds for each agent type.
    - agent_weights (dict): Weight parameters for the agent.
    """
    for iteration in range(max_iterations):
        for i in range(grid_size):
            for j in range(grid_size):
                if is_unsatisfied(grid, i, j, agent_thresholds, agent_weights):
                    move_agent(grid, i, j, strategy)

# Example usage:
grid_size = 20
empty_ratio = 0.2
num_agent_types = 5

# Generate the initial grid with agents and empty cells
grid = generate_general_grid(grid_size, empty_ratio, num_agent_types)

# Assign different similarity thresholds and weights to each agent type
# You can customize these thresholds and weights as needed
agent_thresholds = {agent_type: random.uniform(0.3, 0.6) for agent_type in range(0, num_agent_types + 1)}

# Assign different weights for price and similarity satisfaction
# This reflects the importance that each agent gives to each price and the importance that each type of agent assigns to having similar neighbors
agent_weights = {agent_type: {'weight_price': random.uniform(0.5, 1.0), 'weight_similarity': random.uniform(0.5, 1.0)} for agent_type in range(0, num_agent_types + 1)}

# Define the prices and their probabilities
prices = [1000, 2000, 3000, 4000]
probs_prices = [0.4, 0.3, 0.2, 0.1]

# Generate the grid of prices
grid_prices = generate_price_grid(grid_size, prices, probs_prices)

# Simulate the Schelling model with variable similarity thresholds and weights
max_iterations = 300
strategy = "random"

# Run the Schelling model with the specified parameters
schelling_model(grid, max_iterations, strategy, agent_thresholds, agent_weights)

# Visualize the final state of the grid after simulation
plt.imshow(grid, cmap='cool', vmin=0, vmax=num_agent_types)
plt.colorbar(ticks=list(range(num_agent_types + 1)), label='Agent Type')
plt.title(f'Schelling Model ({max_iterations} iterations, {strategy} strategy)')
plt.show()
