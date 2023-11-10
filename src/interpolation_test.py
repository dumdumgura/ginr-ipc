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


def default_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model-config", type=str, default="./configs/meta_learning/low_rank_modulated_meta/imagenette178_meta_low_rank.yaml")
    #parser.add_argument("-m", "--model-config", type=str,  default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta.yaml")
    parser.add_argument("-m", "--model-config", type=str,default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta_overfit.yaml")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp/")
    parser.add_argument("-t", "--task", type=str, default="lerp_new_13")
    parser.add_argument("-l1", "--load-path_1", type=str, default="./results.tmp/shapenet_meta_overfit/overfit_5_ml13_f128_5/epoch10_model.pt")
    parser.add_argument("-l2", "--load-path_2", type=str, default="./results.tmp/shapenet_meta_overfit/overfit_5_ml13_f128_1/epoch10_model.pt")

    parser.add_argument("-l3", "--load-path_3", type=str,
                        default="./results.tmp/shapenet_meta_overfit/overfit_5_ml13_f128_3/epoch10_model.pt")

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


if __name__ == "__main__":

    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)


    model_1, model_ema_1 = create_model(config.arch, ema=config.arch.ema is not None)
    model_1.to(device)

    # Checkpoint loading
    if not args.load_path_1 == "":
        ckpt_1 = torch.load(args.load_path_1, map_location="cpu")


    if not args.load_path_2 == "":
        ckpt_2 = torch.load(args.load_path_2, map_location="cpu")
        #print(ckpt_2)
    if not args.load_path_3 == "":
        ckpt_3 = torch.load(args.load_path_3, map_location="cpu")

    weight_11 = ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb1']
    weight_21 = ckpt_2['state_dict']['factors.init_modulation_factors.linear_wb1']
    weight_31 = ckpt_3['state_dict']['factors.init_modulation_factors.linear_wb1']

    weight_12 = ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb3']
    weight_22 = ckpt_2['state_dict']['factors.init_modulation_factors.linear_wb3']
    weight_32 = ckpt_3['state_dict']['factors.init_modulation_factors.linear_wb3']


    num_steps = 30
    alphas = np.linspace(0, 1, num_steps)
    meshes=[]
    #interpolated_weights_list = []
    for alpha in alphas:
        break
        interpolated_weights_1 = alpha * weight_11 + (1 - alpha) * weight_21
        interpolated_weights_2 = alpha * weight_12 + (1 - alpha) * weight_22
        #interpolated_weights_list.append(interpolated_weights)
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb1'] = interpolated_weights_1
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb3'] = interpolated_weights_2
        model_1.load_state_dict(ckpt_1["state_dict"], strict=False)
        model_1.to(device)
        xs=1
        coords=3
        mesh = model_1.overfit_one_shape(xs,coords,vis=True)
        reconstruct_shape(mesh,alpha)
    # dist.barrier()

    def generate_point_in_triangle():
        # Generate random barycentric coordinates
        r1, r2 = torch.rand(2)
        sqrt_r1 = torch.sqrt(r1)

        # Calculate barycentric coordinates
        alpha = 1.0 - sqrt_r1
        beta = sqrt_r1 * (1.0 - r2)
        gamma = 1.0 - alpha - beta

        # Calculate the point inside the triangle
        #point = alpha * vertices[0] + beta * vertices[1] + gamma * vertices[2]

        return alpha,beta,gamma

    # High-dimensional triangle vertices (3D for simplicity in this example)
    vertices1= []
    vertices1.append(weight_11)
    vertices1.append(weight_21)
    vertices1.append(weight_31)

    vertices2= []
    vertices2.append(weight_12)
    vertices2.append(weight_22)
    vertices2.append(weight_32)
    for i in range(10):
    # Generate a point inside the triangle
        alpha,beta,gamma= generate_point_in_triangle()
        interpolated_weights_1= alpha * vertices1[0] + beta * vertices1[1] + gamma * vertices1[2]
        interpolated_weights_2 = alpha * vertices2[0] + beta * vertices2[1] + gamma * vertices2[2]
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb1'] = interpolated_weights_1
        ckpt_1['state_dict']['factors.init_modulation_factors.linear_wb3'] = interpolated_weights_2

        model_1.load_state_dict(ckpt_1["state_dict"], strict=False)
        xs = 1
        coords = 3
        mesh = model_1.overfit_one_shape(xs, coords, vis=True)
        reconstruct_shape(mesh, alpha)

    if distenv.master:
        writer.close()
