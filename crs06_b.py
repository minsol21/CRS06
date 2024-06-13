import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 30
NUM_OBJECTS = 300
NUM_ROBOTS = 200
ANTI_AGENT_RATIO = 0.1  # 10% of robots are anti-agents
SIMULATION_STEPS = 5000
PICK_SCALE = 1.5  # Scaling factor for pick-up probability
DROP_SCALE = 2.0  # Scaling factor for drop probability

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Place objects randomly on the grid
for _ in range(NUM_OBJECTS):
    x, y = np.random.randint(0, GRID_SIZE, 2)
    grid[x, y] += 1

# Define robot class
class Robot:
    def __init__(self, is_anti_agent=False):
        self.is_anti_agent = is_anti_agent
        self.x, self.y = np.random.randint(0, GRID_SIZE, 2)
        self.carrying = False

    def move(self):
        self.x = (self.x + np.random.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + np.random.choice([-1, 0, 1])) % GRID_SIZE

    def neighborhood_density(self):
        total_objects = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                nx, ny = (self.x + i) % GRID_SIZE, (self.y + j) % GRID_SIZE
                total_objects += grid[nx, ny]
        return total_objects

    def pick_up(self):
        global grid
        if self.carrying or grid[self.x, self.y] <= 0:
            return
        density = self.neighborhood_density()
        pick_prob = 1 - (1 / (1 + PICK_SCALE * density)) if self.is_anti_agent else (1 / (1 + PICK_SCALE * density))
        if random.random() < pick_prob:
            grid[self.x, self.y] -= 1
            self.carrying = True

    def drop(self):
        global grid
        if not self.carrying:
            return
        density = self.neighborhood_density()
        drop_prob = 1 - (DROP_SCALE * density / (1 + DROP_SCALE * density)) if self.is_anti_agent else (DROP_SCALE * density / (1 + DROP_SCALE * density))
        if random.random() < drop_prob:
            grid[self.x, self.y] += 1
            self.carrying = False

# Initialize robots
robots = [Robot(is_anti_agent=(i < ANTI_AGENT_RATIO * NUM_ROBOTS)) for i in range(NUM_ROBOTS)]

# Simulation loop
for _ in range(SIMULATION_STEPS):
    for robot in robots:
        robot.move()
        if robot.carrying:
            robot.drop()
        else:
            robot.pick_up()

# Visualize the final grid
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Object Clustering after Simulation\nNUM_OBJECTS = {} | NUM_ROBOTS = {}'.format(NUM_OBJECTS, NUM_ROBOTS))
plt.xlabel('Grid Size: {}x{}'.format(GRID_SIZE, GRID_SIZE))
plt.ylabel('Object Density')
plt.show()

# Function to find the largest cluster
def find_largest_cluster(grid):
    visited = np.zeros_like(grid, dtype=bool)
    max_cluster_size = 0

    def dfs(x, y):
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return 0
        if visited[x, y] or grid[x, y] == 0:
            return 0
        visited[x, y] = True
        cluster_size = 1
        cluster_size += dfs(x+1, y)
        cluster_size += dfs(x-1, y)
        cluster_size += dfs(x, y+1)
        cluster_size += dfs(x, y-1)
        return cluster_size

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] > 0 and not visited[i, j]:
                cluster_size = dfs(i, j)
                max_cluster_size = max(max_cluster_size, cluster_size)

    return max_cluster_size

# Run multiple simulations and calculate the average largest cluster size
def run_simulation(num_simulations, anti_agent_ratio):
    global robots, grid
    largest_clusters = []

    for _ in range(num_simulations):
        # Reinitialize grid and robots
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for _ in range(NUM_OBJECTS):
            x, y = np.random.randint(0, GRID_SIZE, 2)
            grid[x, y] += 1
        robots = [Robot(is_anti_agent=(i < anti_agent_ratio * NUM_ROBOTS)) for i in range(NUM_ROBOTS)]

        # Run the simulation
        for _ in range(SIMULATION_STEPS):
            for robot in robots:
                robot.move()
                if robot.carrying:
                    robot.drop()
                else:
                    robot.pick_up()

        # Measure the performance
        largest_clusters.append(find_largest_cluster(grid))

    return np.mean(largest_clusters)

# Test different percentages of anti-agents
anti_agent_ratios = [0.0, 0.05, 0.1, 0.15, 0.2]
num_simulations = 5
performance_results = []

for ratio in anti_agent_ratios:
    avg_largest_cluster = run_simulation(num_simulations, ratio)
    performance_results.append((ratio, avg_largest_cluster))
    print(f"Anti-Agent Ratio: {ratio}, Average Largest Cluster Size: {avg_largest_cluster}")

# Plot the results
ratios, cluster_sizes = zip(*performance_results)
plt.plot(ratios, cluster_sizes, marker='o')
plt.xlabel('Anti-Agent Ratio')
plt.ylabel('Average Largest Cluster Size')
plt.title('Performance of Swarm Clustering with Anti-Agents\nNUM_OBJECTS = {} | NUM_ROBOTS = {}'.format(NUM_OBJECTS, NUM_ROBOTS))
plt.show()
