import open3d as o3d
import numpy as np

# part 1 for checking generated data

# Generate some sample data (replace this with your actual data)
shape_path = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/overfit/2a966a7e0b07a5239a6e43b878d5b335.obj.npy'
data = np.load(shape_path)

points_uniform=128**3
num_points = data[:-points_uniform,3].shape[0]
points = data[:-points_uniform,:3]
bool_values = data[:-points_uniform,3].astype(bool)


# Create Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Set colors based on boolean values
colors = np.zeros((num_points, 3))
colors[bool_values] = [1, 0, 0]  # Set red color for True
colors[~bool_values] = [0, 0, 1]  # Set blue color for False
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])


# part 2 for comparing surface

import open3d as o3d
import numpy as np

path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/3b0efeb0891a9686ca9f0727e23831d9.obj'
# Step 1: Read in the Mesh
mesh = o3d.io.read_triangle_mesh(path)

# Step 2: Sample Points on the Mesh
num_points = 100000  # Adjust the number of points as neede
query = mesh.sample_points_poisson_disk(number_of_points=num_points)
query_points = np.asarray(query.points)

# Create Open3D point cloud
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
#o3d.visualization.draw_geometries([points])

# part 3 for calculating density
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# Assume you already have a point cloud 'point_cloud' with sampled points near the surface

# Convert Open3D point cloud to NumPy array
points = np.asarray(point_cloud.points)

# Set the radius for density estimation
radius = 0.01  # Adjust this value based on your requirements

# Build a k-d tree for efficient nearest neighbor search
kdtree = cKDTree(points)

# Calculate point density for each point
point_densities = kdtree.query_ball_point(query_points, r=radius)
point_density_values = [len(density) for density in point_densities]

# Add point density as colors to the point cloud
density_color = np.array(point_density_values) / max(point_density_values)
zeros = np.zeros(density_color.shape)
query.colors = o3d.utility.Vector3dVector(np.column_stack([density_color, zeros, 1 - density_color]))

print(np.array(point_density_values).mean())
# Visualize the point cloud with density information
o3d.visualization.draw_geometries([query])
