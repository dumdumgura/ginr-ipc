import os
import time

import open3d as o3d
import numpy as np
import igl
import matplotlib.pyplot as plt

#strategy = 'sdf'
strategy = 'occ'


folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/val_obj'
save_folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/val_overfit'

filter = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/shape_filter_small.txt'

files = []
filter_list = []
with open(filter, "r") as text:
    for line in text:
        file_name = line.strip()
        filter_list.append(file_name)

for file in os.listdir(folder):
    if file in filter_list:
        files.append(file)

i=0


# Step 1: Read in the Mesh
for file in files:
    mesh = o3d.io.read_triangle_mesh(os.path.join(folder,file))

    vertices = mesh.vertices
    vertices -= np.mean(vertices, axis=0, keepdims=True)

    v_max = np.amax(vertices)
    v_min = np.amin(vertices)
    vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #self.obj = obj

    #total_points =  1000000
    n_points_uniform = 30000 # int(total_points * 0.5)
    n_points_surface = 40000 # total_points

    n_points_surface = 1000000
    n_points_uniform = 1000000


    if strategy == 'occ':
        points_uniform = np.random.uniform(
            -0.5, 0.5, size=(n_points_uniform, 3)
        )

        min_bound = np.array([-0.5,-0.5,-0.5])
        max_bound = np.array([0.5, 0.5, 0.5])
        xyz_range = np.linspace(min_bound, max_bound, num=128)

        grid = np.meshgrid(*xyz_range.T)
        points_uniform = np.stack(grid, axis=-1).astype(np.float32)
        points_uniform = points_uniform.reshape((-1,3))

        start = time.time()
        print('sampling near surface...')
        points_surface = np.asarray(mesh.sample_points_poisson_disk(number_of_points=n_points_surface).points)
        end = time.time()
        print('sampling takes: '+str(end-start))


        res = 0.0001
        points_surface_1 = points_surface + 0.001 * np.random.randn(n_points_surface, 3)
        points_surface_2 = points_surface + 0.01 * np.random.randn(n_points_surface, 3)
        points_surface_3 = points_surface + 0.0001 * np.random.randn(n_points_surface, 3)
        #points_surface_4 = points_surface + 0.00001 * np.random.randn(n_points_surface, 3)


        points = np.concatenate([points_surface_1,points_surface_2,points_surface_3, points_uniform], axis=0)
        labels = np.zeros(points.shape[0])
        labels[:-n_points_uniform] = 1 # 1 means near surface



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



        #print(points.shape, occupancies.shape, occupancies.sum())

        point_cloud = points
        point_cloud = np.hstack((point_cloud, occupancies, labels[:,None]))
        print(point_cloud.shape, points.shape, occupancies.shape)
        print(i)
        i=i+1
        save_path = os.path.join(save_folder,file)
        np.save(save_path,point_cloud)

    else:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

        min_bound = mesh.vertex.positions.min(0).numpy()
        max_bound = mesh.vertex.positions.max(0).numpy()

        min_bound = np.array([-0.5,-0.5,-0.5])
        max_bound = np.array([0.5, 0.5, 0.5])
        xyz_range = np.linspace(min_bound, max_bound, num=64)

        # query_points is a [32,32,32,3] array ..
        grid = np.meshgrid(*xyz_range.T)
        query_points = np.stack(grid, axis=-1).astype(np.float32)


        start = time.time()
        print('sampling near surface...')
        points_surface = np.asarray(mesh.sample_points_poisson_disk(number_of_points=n_points_surface).points)
        end = time.time()
        print('sampling takes: '+str(end-start))
        res = 0.0001
        points_surface_1 = points_surface + 0.001 * np.random.randn(n_points_surface, 3)
        points_surface_2 = points_surface + 0.01 * np.random.randn(n_points_surface, 3)


        # signed distance is a [32,32,32] array
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        occupancy = scene.compute_occupancy(query_points).numpy()

        print(str(signed_distance.shape)+"_"+str(occupancy.shape))

        sdf_volume = np.concatenate([query_points,signed_distance[:,:,:,None],occupancy[:,:,:,None]],axis=-1)
        print(sdf_volume.shape)

        sdf_data= sdf_volume.reshape((-1,5))

        # We can visualize a slice of the distance field directly with matplotlib
        #plt.imshow(signed_distance.numpy()[16, : , :])


