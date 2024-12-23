# K-Means Clustering Visualization

This project demonstrates the K-Means clustering algorithm using Python's NumPy and Matplotlib libraries. The algorithm partitions a set of randomly generated data points into a specified number of clusters and visualizes the clustering process step-by-step through an animated plot.

## Features

- **Random Data Generation**: Generates a user-defined number of random data points within a specified range.
- **K-Means Algorithm**: Implements the core steps of the K-Means algorithm: point assignment to clusters and centroid recalculation.
- **Visualization**: Displays the clustering process as an animation with updated point assignments and centroid positions at each iteration.
- **Customization**: Allows users to configure parameters like the number of clusters, animation delay, point size, and cluster colors.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

- `numpy`
- `matplotlib`

You can install them using pip if they are not already installed:

```bash
pip install numpy matplotlib
```

### Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/aditimishra28/KMeansAlgo.git
   ```

2. Navigate to the project directory:
   ```bash
   cd KMeansAlgo
   ```

3. Run the Python script:
   ```bash
   python KMeansAlgo.py
   ```

4. The animation will display, showing the clustering process.

## Parameters

You can modify the following parameters in the script to customize the behavior:

- **`num_points`**: Number of data points to generate.
- **`grid_size`**: Maximum range for randomly generated points.
- **`num_clusters`**: Number of clusters (k) for the algorithm.
- **`animation_delay`**: Time delay between frames in the animation (milliseconds).
- **`point_size`**: Size of the points in the scatter plot.
- **`centroid_size`**: Size of the centroids in the scatter plot.
- **`cluster_colors`**: List of colors for each cluster.

## Code Structure

- **Data Generation**: Random points are generated using `numpy`.
- **Initialization**: Cluster centroids are initialized by selecting random points.
- **Point Assignment**: Each point is assigned to the nearest centroid.
- **Centroid Update**: Centroids are recalculated as the mean of points in each cluster.
- **Animation**: Matplotlib's `FuncAnimation` is used to visualize the clustering process.

## Example Output

The animation shows the following:
- Points being assigned to clusters.
- Centroids moving to the mean of their respective clusters.
- The process stopping once centroids stabilize.



