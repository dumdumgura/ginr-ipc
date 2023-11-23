from mesh_to_sdf import mesh_to_voxels
import time
import trimesh
import skimage


from mesh_to_sdf import sample_sdf_near_surface
import pyrender
import numpy as np
import os

'''
path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/1a6ad7a24bb89733f412783097373bdc.obj'
mesh = trimesh.load(path)
start = time.time()
voxels = mesh_to_voxels(mesh, 32, pad=True,sign_method='depth')

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.01)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
end =time.time()
print(end-start)
#mesh.show()
folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/obj'
save_folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_test/npy'
'''


folder = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/'
save_folder = '/home/umaru/praktikum/changed_version/ginr-ipc/data/shapenet/sdf_pc_30w'


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
    start = time.time()
    mesh = trimesh.load(os.path.join(folder,file))

    #vertices = mesh.vertices
    #vertices -= np.mean(vertices, axis=0, keepdims=True)

    #v_max = np.amax(vertices)
    #v_min = np.amin(vertices)
    #vertices *= 0.5 * 0.95 / (max(abs(v_min), abs(v_max)))

    #mesh.vertices = vertices


#path = '/home/umaru/praktikum/changed_version/HyperDiffusion/data/02691156_manifold/1a6ad7a24bb89733f412783097373bdc.obj'
#mesh = trimesh.load(path)
    points, sdf, grads = sample_sdf_near_surface(mesh, number_of_points=300000,sign_method='depth',return_gradients=True)#1000w

    point_cloud = np.concatenate([points.reshape(-1, 3), sdf.reshape(-1,1),grads.reshape(-1,3)], axis=-1)


    print(point_cloud.shape)
    #save_path = os.path.join(save_folder, 'test')
    print(i)
    i = i + 1
    save_path = os.path.join(save_folder, file)
    np.save(save_path, point_cloud)
    end =time.time()
    print(end-start)


    print(np.count_nonzero(sdf>0.0))
    print(np.count_nonzero(sdf<0.0))

    '''

    colors = np.zeros(points.shape)
    colors[sdf < 0.00, 2] = 1
    colors[sdf > 0.00, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)


    #scaled_normal = 0.01 * grads/np.linalg.norm(grads,axis=-1,keepdims=True)
    #line_set=[]
        # Create LineSet for the point and its normal
   #for points,normal in zip(points,scaled_normal):

    #   line = pyrender.Primitive(positions=np.array([points, points+normal]),mode=1)
    #line_set.positions = points
    #line_set.normals = scaled_normal
    # Add the LineSet to the scene
    #    line_set.append(line)
        #break
    #node = pyrender.Mesh(primitives=line_set)
    # Add the Node to the scene
    #scene.add_node(node)
    #scene.add(node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=3)

    '''