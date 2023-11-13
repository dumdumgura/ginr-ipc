import argparse
import math

import torch
import torch.distributed as dist
import numpy as np
import utils.dist as dist_utils
import plyfile
from models import create_model
from trainers import create_trainer, STAGE_META_INR_ARCH_TYPE
from datasets import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup
import time
from skimage import measure
import open3d as o3d

def default_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model-config", type=str, default="./configs/meta_learning/low_rank_modulated_meta/imagenette178_meta_low_rank.yaml")
    #parser.add_argument("-m", "--model-config", type=str,  default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta.yaml")
    parser.add_argument("-m", "--model-config", type=str,default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta_overfit.yaml")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp/")
    parser.add_argument("-t", "--task", type=str, default="lerp_new_29")
    parser.add_argument("-l1", "--load-path_1", type=str, default="./results.tmp/shapenet_meta_overfit/overfit_ml13_occ_128_reweight_100_10/epoch400_model.pt")
    parser.add_argument("-l2", "--load-path_2", type=str, default="./results.tmp/shapenet_meta_overfit/overfit_ml13_occ_128_reweight_100_11/epoch500_model.pt")

    #parser.add_argument("-l3", "--load-path_3", type=str,
    #                    default="./results.tmp/shapenet_meta_overfit/overfit_5_ml13_f128_3/epoch10_model.pt")

    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser


def add_dist_arguments(parser):
    parser.add_argument("--world_size", default=0, type=int, help="number of nodes for distributed training")
    parser.add_argument("--local_rank", default=1, type=int, help="local rank for distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--nproc_per_node", default=1, type=int)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--timeout", type=int, default=1, help="time limit (s) to wait for other nodes in DDP")
    return parser


def parse_args():
    parser = default_parser()
    parser = add_dist_arguments(parser)
    args, extra_args = parser.parse_known_args()
    return args, extra_args

class MultiGridExtractor(object):
    def __init__(self, resolution0, threshold):
        # Attributes
        self.resolution = resolution0
        self.threshold = threshold

        # Voxels are active or inactive,
        # values live on the space between voxels and are either
        # known exactly or guessed by interpolation (unknown)
        shape_voxels = (resolution0,) * 3
        shape_values = (resolution0 + 1,) * 3
        self.values = np.empty(shape_values,dtype=np.float32)
        self.value_known = np.full(shape_values, False)
        self.voxel_active = np.full(shape_voxels, True)

    def query(self):
        # Query locations in grid that are active but unkown
        idx1, idx2, idx3 = np.where(
            ~self.value_known & self.value_active
        )
        points = np.stack([idx1, idx2, idx3], axis=-1)
        return points

    def update(self, points, values):
        # Update locations and set known status to true
        idx0, idx1, idx2 = points.transpose()
        self.values[idx0, idx1, idx2] = values
        self.value_known[idx0, idx1, idx2] = True

        # Update activity status of voxels accordings to new values
        self.voxel_active = ~self.voxel_empty
        # (
        #     # self.voxel_active &
        #     self.voxel_known & ~self.voxel_empty
        # )

    def increase_resolution(self):
        self.resolution = 2 * self.resolution
        shape_values = (self.resolution + 1,) * 3

        value_known = np.full(shape_values, False)
        value_known[::2, ::2, ::2] = self.value_known
        values = upsample3d_nn(self.values)
        values = values[:-1, :-1, :-1]

        self.values = values
        self.value_known = value_known
        self.voxel_active = upsample3d_nn(self.voxel_active)


    @property
    def occupancies(self):
        return (self.values < self.threshold)

    @property
    def value_active(self):
        value_active = np.full(self.values.shape, False)
        # Active if adjacent to active voxel
        value_active[:-1, :-1, :-1] |= self.voxel_active
        value_active[:-1, :-1, 1:] |= self.voxel_active
        value_active[:-1, 1:, :-1] |= self.voxel_active
        value_active[:-1, 1:, 1:] |= self.voxel_active
        value_active[1:, :-1, :-1] |= self.voxel_active
        value_active[1:, :-1, 1:] |= self.voxel_active
        value_active[1:, 1:, :-1] |= self.voxel_active
        value_active[1:, 1:, 1:] |= self.voxel_active

        return value_active

    @property
    def voxel_known(self):
        value_known = self.value_known
        voxel_known = self.check_voxel_occupied(value_known)
        return voxel_known

    @property
    def voxel_empty(self):
        occ = self.occupancies
        return ~self.check_voxel_boundary(occ)

    def check_voxel_occupied(self,occupancy_grid):
        occ = occupancy_grid

        occupied = (
                occ[..., :-1, :-1, :-1]
                & occ[..., :-1, :-1, 1:]
                & occ[..., :-1, 1:, :-1]
                & occ[..., :-1, 1:, 1:]
                & occ[..., 1:, :-1, :-1]
                & occ[..., 1:, :-1, 1:]
                & occ[..., 1:, 1:, :-1]
                & occ[..., 1:, 1:, 1:]
        )
        return occupied

    def check_voxel_unoccupied(self,occupancy_grid):
        occ = occupancy_grid

        unoccupied = ~(
                occ[..., :-1, :-1, :-1]
                | occ[..., :-1, :-1, 1:]
                | occ[..., :-1, 1:, :-1]
                | occ[..., :-1, 1:, 1:]
                | occ[..., 1:, :-1, :-1]
                | occ[..., 1:, :-1, 1:]
                | occ[..., 1:, 1:, :-1]
                | occ[..., 1:, 1:, 1:]
        )
        return unoccupied

    def check_voxel_boundary(self,occupancy_grid):
        occupied = self.check_voxel_occupied(occupancy_grid)
        unoccupied = self.check_voxel_unoccupied(occupancy_grid)
        return ~occupied & ~unoccupied


    def extract_mesh(self):
        # Ensure that marching cubes is applied to the current resolution
        occupancies = self.occupancies
        vertices, faces, _, _ = measure.marching_cubes(occupancies, level=self.threshold)

        # You may want to perform additional processing on the mesh
        # For example, simplify and refine the mesh using gradient information

        return vertices, faces


def upsample3d_nn(x):
    xshape = x.shape
    yshape = (2*xshape[0], 2*xshape[1], 2*xshape[2])

    y = np.zeros(yshape, dtype=x.dtype)
    y[::2, ::2, ::2] = x
    y[::2, ::2, 1::2] = x
    y[::2, 1::2, ::2] = x
    y[::2, 1::2, 1::2] = x
    y[1::2, ::2, ::2] = x
    y[1::2, ::2, 1::2] = x
    y[1::2, 1::2, ::2] = x
    y[1::2, 1::2, 1::2] = x

    return y




def reconstruct_shape(meshes,alpha,it=1,mode='lerp'):
    for k in range(len(meshes)):
        # try writing to the ply file
        verts = meshes[k]['vertices']
        faces = meshes[k]['faces']
        voxel_grid_origin = [-0.5] * 3
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        # logging.debug("saving mesh to %s" % (ply_filename_out))
        ply_data.write("./results.tmp/ply/" + str(alpha) + "_" +str(mode)+"_"+ str(it*len(meshes)+k) + "_poly.ply")


def reconstruct_shape_with_filtering(meshes,epoch,alpha,mode='lerp'):

    for k in range(len(meshes)):
        # try writing to the ply file
        verts = meshes[k]['vertices']
        faces = meshes[k]['faces']

        sample_mesh = o3d.geometry.TriangleMesh()

        sample_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        sample_mesh.triangles = o3d.utility.Vector3iVector(faces)
        # Compute normals for the mesh
        sample_mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([sample_mesh])

        # filter out small blobs:
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (sample_mesh.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        print("Show mesh with small clusters removed")

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 20
        sample_mesh.remove_triangles_by_mask(triangles_to_remove)
        # o3d.visualization.draw_geometries([mesh_0])

        print('filter with average with 5 iterations')
        iter = int(epoch / 40)
        mesh_out = sample_mesh.filter_smooth_simple(number_of_iterations=5)
        mesh_out.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh_out])

        verts, faces = np.asarray(mesh_out.vertices), np.asarray(mesh_out.triangles)

        voxel_grid_origin = [-0.5] * 3
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        # logging.debug("saving mesh to %s" % (ply_filename_out))
        ply_data.write(
            "./results.tmp/ply/" + str(epoch) + "_" + str(mode) + "_" + str(alpha) + "_poly.ply")


if __name__ == "__main__":

    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    strategy='adaptive'
    #strategy='fix'

    model_1, model_ema_1 = create_model(config.arch, ema=config.arch.ema is not None)
    model_1.to(device)

    # Checkpoint loading
    if not args.load_path_1 == "":
        ckpt_1 = torch.load(args.load_path_1, map_location="cpu")


    if not args.load_path_2 == "":
        ckpt_2 = torch.load(args.load_path_2, map_location="cpu")
        #print(ckpt_2)

    weight_11 = ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb1']
    weight_21 = ckpt_2['state_dict']['factors.init_modulation_factors.linear_wb1']

    weight_12 = ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb3']
    weight_22 = ckpt_2['state_dict']['factors.init_modulation_factors.linear_wb3']

    num_steps = 10
    alphas = np.linspace(0, 1, num_steps)
    meshes=[]
    #interpolated_weights_list = []
    for alpha in alphas:

        interpolated_weights_1 = alpha * weight_11 + (1 - alpha) * weight_21
        interpolated_weights_2 = alpha * weight_12 + (1 - alpha) * weight_22
        #interpolated_weights_list.append(interpolated_weights)
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb1'] = interpolated_weights_1
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb3'] = interpolated_weights_2
        model_1.load_state_dict(ckpt_1["state_dict"], strict=False)
        model_1.to(device)
        if strategy != 'adaptive':
            xs=1
            coords=3
            mesh = model_1.overfit_one_shape(xs,coords,vis=True)
            reconstruct_shape(mesh,alpha)
        else:
            # Example Usage
            resolution0 = 80
            final_res = 1024
            res = resolution0

            threshold = 0.0
            multi_grid_extractor = MultiGridExtractor(resolution0, threshold)

            for i in range(4):
                start = time.time()
                # Initial query to get points for occupancy evaluation
                overall_index = multi_grid_extractor.query()
                num_samples = overall_index.shape[0]  # (10+1) ** 3

                # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
                voxel_origin = [-0.5] * 3
                voxel_size = -2 * voxel_origin[0] / (res)

                samples = np.zeros((num_samples, 3))

                # transform first 3 columns
                # to be the x, y, z index
                # samples[:, 2] = overall_index % res
                # samples[:, 1] = (overall_index.long() / res) % res
                # samples[:, 0] = ((overall_index.long() / res) / res) % res

                # transform first 3 columns
                # to be the x, y, z coordinate
                samples[:, 0] = (overall_index[:, 0] * voxel_size) + voxel_origin[2]
                samples[:, 1] = (overall_index[:, 1] * voxel_size) + voxel_origin[1]
                samples[:, 2] = (overall_index[:, 2] * voxel_size) + voxel_origin[0]

                # Simulate evaluating occupancy at the queried points
                # For simplicity, let's assume random values for illustration
                head = 0
                max_batch = 1024
                values_at_points = np.zeros((num_samples, 1))

                while head < num_samples:
                    # print(head)

                    sample_subset = torch.tensor(samples[head: min(head + max_batch, num_samples), 0:3],
                                                 dtype=torch.float).cuda()
                    values_at_points[head: min(head + max_batch, num_samples), 0] = (
                        model_1.overfit_one_shape(xs=1, coord=sample_subset).squeeze().detach().cpu()
                        # .squeeze(1)
                    )

                    head += max_batch

                # occ_values = values_at_points.reshape(res+1,res+1,res+1)
                multi_grid_extractor.update(overall_index.squeeze(), values_at_points.squeeze())

                # After increasing resolution, you might perform additional queries and updates
                # ...

                # Increase resolution
                if i !=3:
                    multi_grid_extractor.increase_resolution()
                    res = res * 2


            # Extract the mesh using marching cubes
            vertices, faces = multi_grid_extractor.extract_mesh()
            meshes = []
            tmp = {}
            tmp['vertices'], tmp['faces'] = vertices, faces
            meshes.append(tmp)
            tmp = {}
            reconstruct_shape_with_filtering(meshes, res,alpha)
            print(res)
            end = time.time()
            print("sampling takes: %f" % (end - start))

    # dist.barrier()

    if distenv.master:
        writer.close()
