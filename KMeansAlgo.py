import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for K-Means clustering
num_points = 2000         # Number of data points to generate
grid_size = 20000         # Maximum range for randomly generated points
num_clusters = 10       # Number of clusters (k)
animation_delay = 100     # Time delay between frames in the animation (ms)
point_size = 5            # Size of the points in the scatter plot
centroid_size = 100       # Size of the centroids in the scatter plot
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'lime', 'pink']  # Colors for each cluster

# Step 1: Generate random data points
data_points = np.random.rand(num_points, 2) * grid_size

# Step 2: Initialize cluster centroids by selecting random points
initial_centroids = np.random.choice(len(data_points), num_clusters, replace=False)
cluster_centroids = data_points[initial_centroids]
point_assignments = np.zeros(num_points, dtype=int)  # Assign each point to a cluster
iteration_counter = 0  # Keep track of the number of iterations

# Helper function to assign points to the nearest cluster
def assign_points_to_clusters(points, centroids):
    """
    Assign each data point to the nearest cluster centroid.
    Returns:
        - cluster_labels: An array of cluster indices for each point.
        - distances: The distances of points to their assigned centroids.
    """
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels, np.min(distances, axis=1)

# Helper function to update cluster centroids
def recalculate_centroids(points, assignments, num_clusters):
    """
    Compute the new centroids as the mean of points in each cluster.
    """
    updated_centroids = np.zeros((num_clusters, 2))
    for cluster_id in range(num_clusters):
        cluster_points = points[assignments == cluster_id]
        if len(cluster_points) > 0:  # Avoid division by zero
            updated_centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return updated_centroids

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size * 1.1)

# Scatter plot for points and centroids
scatter_plot = ax.scatter([], [], s=point_size, alpha=0.7)
centroid_plot = ax.scatter([], [], color='black', s=centroid_size, label='Centroids')

# Display the iteration count in the title
plot_title = ax.set_title(f"K-Means Clustering - Iteration {iteration_counter}")

# Function to update the plot for each animation frame
def update_plot(frame):
    global cluster_centroids, point_assignments, iteration_counter

    # Step 1: Assign points to the nearest cluster
    point_assignments, _ = assign_points_to_clusters(data_points, cluster_centroids)

    # Step 2: Recalculate cluster centroids
    new_centroids = recalculate_centroids(data_points, point_assignments, num_clusters)

    # Update scatter plot colors and centroid positions
    scatter_colors = [cluster_colors[cluster] for cluster in point_assignments]
    scatter_plot.set_offsets(data_points)
    scatter_plot.set_color(scatter_colors)
    centroid_plot.set_offsets(new_centroids)

    # Update the iteration count in the title
    iteration_counter += 1
    plot_title.set_text(f"K-Means Clustering - Iteration {iteration_counter}")

    # Stop the animation if centroids have stabilized
    if np.allclose(cluster_centroids, new_centroids, atol=1e-4):
        anim.event_source.stop()

    # Update centroids for the next iteration
    cluster_centroids[:] = new_centroids

# Create the animation
anim = animation.FuncAnimation(fig, update_plot, interval=animation_delay, frames=100, repeat=False)

# Show the final result
plt.legend()
plt.show()



