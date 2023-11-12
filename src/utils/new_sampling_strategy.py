
import open3d as o3d
import numpy as np
import igl

path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/3b0efeb0891a9686ca9f0727e23831d9.obj'
# Step 1: Read in the Mesh
mesh = o3d.io.read_triangle_mesh(path)

vertices = mesh.vertices
vertices -= np.mean(vertices, axis=0, keepdims=True)

v_max = np.amax(vertices)
v_min = np.amin(vertices)
vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))

mesh.vertices = o3d.utility.Vector3dVector(vertices)
#self.obj = obj

total_points =  200000
n_points_uniform = total_points # int(total_points * 0.5)
n_points_surface = total_points  # total_points

points_uniform = np.random.uniform(
    -0.5, 0.5, size=(n_points_uniform, 3)
)

points_surface = np.asarray(mesh.sample_points_poisson_disk(number_of_points=total_points).points)



points_surface_1 = points_surface + 0.001 * np.random.randn(n_points_surface, 3)
points_surface_2 = points_surface + 0.01 * np.random.randn(n_points_surface, 3)
points_surface_3 = points_surface + 0.0001 * np.random.randn(n_points_surface, 3)



points = np.concatenate([points_surface_1,points_surface_2,points_surface_3, points_uniform], axis=0)


inside_surface_values = igl.fast_winding_number_for_meshes(
    np.asarray(mesh.vertices), np.asarray(mesh.triangles), points
)

thresh = 0.5

occupancies_winding = np.piecewise(
    inside_surface_values,
    [inside_surface_values < thresh, inside_surface_values >= thresh],
    [0, 1],
)
occupancies = occupancies_winding[..., None]





print(points.shape, occupancies.shape, occupancies.sum())

point_cloud = points
point_cloud = np.hstack((point_cloud, occupancies))
print(point_cloud.shape, points.shape, occupancies.shape)

np.save('/home/umaru/praktikum/changed_version/ginr-ipc/data/test_data/shape',point_cloud)