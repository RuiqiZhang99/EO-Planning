import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Randomly generate points
def generate_points(num_points, x_range, y_range, z_range):
    points = np.empty((num_points, 3))
    points[:, 0] = np.random.rand(num_points) * x_range
    points[:, 1] = np.random.rand(num_points) * y_range
    points[:, 2] = np.random.rand(num_points) * z_range
    return points

# Calculate physical distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Calculate energy consumption between two points
def calculate_energy(point1, point2):
    diff = point1 - point2
    xy_energy = np.sqrt(diff[0]**2 + diff[1]**2)
    if diff[2] > 0:
        z_energy = diff[2] * 10  # Ascending
    else:
        z_energy = 0  # Descending
    return xy_energy + z_energy

# Greedy algorithm to find the shortest distance path
def greedy_algorithm_distance(points):
    num_points = len(points)
    visited = [False] * num_points
    path = [0]
    visited[0] = True
    
    for _ in range(1, num_points):
        last_point = path[-1]
        next_point = None
        min_distance = float('inf')
        
        for i in range(num_points):
            if not visited[i]:
                dist = calculate_distance(points[last_point], points[i])
                if dist < min_distance:
                    min_distance = dist
                    next_point = i
        
        path.append(next_point)
        visited[next_point] = True
    
    # Return to the starting point to form a loop
    path.append(0)
    return path

# Greedy algorithm to find the minimum energy path
def greedy_algorithm_energy(points):
    num_points = len(points)
    visited = [False] * num_points
    path = [0]
    visited[0] = True
    
    for _ in range(1, num_points):
        last_point = path[-1]
        next_point = None
        min_energy = float('inf')
        
        for i in range(num_points):
            if not visited[i]:
                energy = calculate_energy(points[last_point], points[i])
                if energy < min_energy:
                    min_energy = energy
                    next_point = i
        
        path.append(next_point)
        visited[next_point] = True
    
    # Return to the starting point to form a loop
    path.append(0)
    return path

# Calculate total energy consumption of a path
def calculate_total_energy(path, points):
    return sum(calculate_energy(points[path[i]], points[path[i+1]]) for i in range(len(path) - 1))

# Add arrow at the midpoint between two waypoints to indicate path direction
def add_arrow(ax, point1, point2, color):
    arrow = (point2 - point1) / 10
    midpoint = (point1 + point2) / 2
    ax.quiver(midpoint[0], midpoint[1], midpoint[2], arrow[0], arrow[1], arrow[2], color=color)

# Parameters
num_points = 10
x_range = 100
y_range = 100
z_range = 200

# Generate points
points = generate_points(num_points, x_range, y_range, z_range)

# Run greedy algorithms
distance_path_indices = greedy_algorithm_distance(points)
energy_path_indices = greedy_algorithm_energy(points)

distance_path = [points[i] for i in distance_path_indices]
energy_path = [points[i] for i in energy_path_indices]

# Calculate energy consumption of paths
distance_path_energy = calculate_total_energy(distance_path_indices, points)
energy_path_energy = calculate_total_energy(energy_path_indices, points)

# Print paths and energy consumption
print("Shortest Distance Path Waypoints:", distance_path_indices)
print("Shortest Distance Path Total Energy Consumption:", distance_path_energy)
print("Minimum Energy Path Waypoints:", energy_path_indices)
print("Minimum Energy Path Total Energy Consumption:", energy_path_energy)


# 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black')

# Plot shortest distance path
distance_path_points = np.array(distance_path)
ax.plot(distance_path_points[:, 0], distance_path_points[:, 1], distance_path_points[:, 2], c='red', label='Shortest Distance Path', alpha=0.5)

# Add arrows to shortest distance path
for i in range(len(distance_path_points) - 1):
    add_arrow(ax, distance_path_points[i], distance_path_points[i+1], 'red')

# Plot minimum energy path
energy_path_points = np.array(energy_path)
ax.plot(energy_path_points[:, 0], energy_path_points[:, 1], energy_path_points[:, 2], c='blue', label='Minimum Energy Path', alpha=0.5)

# Add arrows to minimum energy path
for i in range(len(energy_path_points) - 1):
    add_arrow(ax, energy_path_points[i], energy_path_points[i+1], 'blue')

# Annotate points
for i, point in enumerate(points):
    ax.text(point[0], point[1], point[2], '%s' % (str(i)), size=20, zorder=1, color='k')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
plt.show()