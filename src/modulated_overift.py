import argparse
import math

import torch
import torch.distributed as dist

import utils.dist as dist_utils

from models import create_model
from trainers import create_trainer, STAGE_META_INR_ARCH_TYPE
from datasets import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup


def default_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-m", "--model-config", type=str, default="./configs/meta_learning/low_rank_modulated_meta/imagenette178_meta_low_rank.yaml")
    parser.add_argument("-m", "--model-config", type=str,default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta.yaml")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp/")
    parser.add_argument("-t", "--task", type=str, default="test_bs4_nl6_hd_256_nil4_adlr46_sl1_factor128_data266")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true",default=False)
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

def main():
    path_to_ckpt = "/home/umur/Documents/TUM/praktikum/ginr-ipc/results.tmp/shapenet_meta/test_bs4_nl6_hd_256_nil4_adlr46_sl1_factor128_data266/epoch2_model.pt"
    #checkpoint = torch.load(path_to_ckpt)
    #print(checkpoint)
    
    
    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, _ = create_dataset(config, is_eval=args.eval, logger=logger)

    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    model_ema = model_ema.to(device) if model_ema is not None else None

    if distenv.master:
        print(model)
        profiler.get_model_size(model)
        profiler.get_model_size(model, opt="trainable-only")

    # Checkpoint loading
    ckpt = torch.load(path_to_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    if model_ema is not None:
        model_ema.module.load_state_dict(ckpt["state_dict_ema"])

    if distenv.master:
        logger.info(f"{args.load_path} model is loaded")
        
        
    # Optimizer definition
    steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
    steps_per_epoch = steps_per_epoch // config.optimizer.grad_accm_steps

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(
        optimizer, config.optimizer.warmup, steps_per_epoch, config.experiment.epochs_cos, distenv
    )
    
    if distenv.master:
        print(optimizer)

    # Set trainer
    trainer = create_trainer(config)
    trainer = trainer(model, model_ema, dataset_trn, None, config, writer, device, distenv)

    epoch_st = 0
    trainer.run_epoch(optimizer, scheduler, epoch_st)
    
if __name__ == "__main__":
    main()