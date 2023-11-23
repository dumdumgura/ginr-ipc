import logging
import time

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import plyfile

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate


logger = logging.getLogger(__name__)


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        n_inner_step = self.config.arch.n_inner_step
        accm = AccmStageINR(
            scalar_metric_names=("loss_total", "mse", "psnr","onsurface_loss","spatial_loss","grad_loss","normal_loss","div_loss","bce_loss"),
            vector_metric_names=("inner_mse", "inner_psnr"),
            vector_metric_lengths=(n_inner_step, n_inner_step),
            device=self.device,
        )
        return accm

    @torch.no_grad()

    def reconstruct_shape(self,meshes,epoch,it=0,mode='train'):
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
            ply_data.write("./results.tmp/ply/" + str(epoch) + "_" +str(mode)+"_"+ str(it*len(meshes)+k) + "_poly.ply")

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xt in pbar:
            model.zero_grad()
            if self.config.dataset.type == "shapenet":

                coord_inputs = xt['coords'].to(self.device)
                coord_inputs.requires_grad_()
                labels=xt['label'].to(self.device)
                if self.config.dataset.supervision == 'sdf':
                    xs = xt['sdf'].to(self.device)
                    normals = xt['normal'].to(self.device)
                    xs = torch.concatenate([xs,normals],dim=-1)
                else:
                    xs = xt['occ'].to(self.device)

            else:
                xs = xt.to(self.device)
                coord_inputs = model.sample_coord_input(xs, device=xs.device)


            vis = False
            if self.config.dataset.type == 'shapenet':
                outputs, _, collated_history = model(xs, coord_inputs, is_training=False,vis=vis)
                #self.reconstruct_shape(meshes,epoch,it)
            else:
                outputs, _, collated_history = model(xs, coord_inputs, is_training=False, vis=vis)


            targets = xs.detach()

            #loss = model.module.compute_loss(outputs, targets, reduction="sum")
            #loss = model.compute_loss(outputs, targets, reduction="ce",label=labels)
            loss = model.compute_loss(outputs, targets,reduction="ce",label=labels,type=self.config.dataset.supervision,coords=coord_inputs,mode='sum')


            metrics = dict(
                loss_total=loss["loss_total"],
                #mse=loss["mse"],
                #psnr=loss["psnr"],

                onsurface_loss = loss["onsurface_loss"],
                spatial_loss = loss["spatial_loss"],
                grad_loss=loss["grad_loss"],
                normal_loss=loss["normal_loss"],
                div_loss=loss["div_loss"],
                bce_loss=loss["bce_loss"]
                #inner_mse=collated_history["mse"],
                #inner_psnr=collated_history["psnr"],
            )
            accm.update(metrics, count=xs.shape[0], sync=True, distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)

        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%sudo apt install python3.7s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            #self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["xs"] = xt

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        start = time.time()
        model = self.model
        model_ema = self.model_ema
        total_step = len(self.loader_trn) * epoch

        accm = self.get_accm()
        #self.distenv.master = 0
        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
            #pbar = enumerate(self.loader_trn)
        else:
            pbar = enumerate(self.loader_trn)

        print()
        model.train()
        end_2 = time.time()
        print(end_2 - start)
        for it, xt in pbar:
            end = time.time()
            print(end-start)
            model.zero_grad(set_to_none=True)
            #xs = xs.to(self.device, non_blocking=True)

            if self.config.dataset.type == "shapenet":

                coord_inputs = xt['coords'].to(self.device)
                coord_inputs.requires_grad_()
                labels=xt['label'].to(self.device)

                if self.config.dataset.supervision == 'sdf':
                    xs = xt['sdf'].to(self.device)
                    normals = xt['normal'].to(self.device)
                    xs = torch.concatenate([xs,normals],dim=-1)
                else:
                    xs = xt['occ'].to(self.device)


            else:
                xs = xt.to(self.device)
                coord_inputs = model.sample_coord_input(xs, device=xs.device)

            #coord_inputs = model.module.sample_coord_input(xs, device=xs.device)
            #coord_inputs = model.sample_coord_input(xs, device=xs.device)
            #prediction


            if  self.config.type == 'overfit':
                outputs = model.overfit_one_shape(xs, coord=coord_inputs)
            else:
                outputs, _, collated_history = model(xs, coord=coord_inputs, is_training=True,label=labels,type=self.config.dataset.supervision)

            targets = xs.detach()
            #loss = model.module.compute_loss(outputs, targets)


            if self.config.arch.hyponet.fourier_mapping in ['deterministic_transinr']:
                loss_type = 'mean'
                label = None
            else:
                loss_type = 'ce'
                label = labels

            loss = model.compute_loss(outputs, targets,reduction=loss_type,label=labels,type=self.config.dataset.supervision,coords=coord_inputs)

            epoch_loss =float(loss["loss_total"].item())

            loss["loss_total"].backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optimizer.max_gn)
            optimizer.step()

            if scheduler.mode =='adaptive':

                scheduler.step(epoch_loss)
            else:
                scheduler.step(epoch)

            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(
                loss_total=loss["loss_total"],
                #mse=loss["mse"],
                #psnr=loss["psnr"],

                onsurface_loss = loss["onsurface_loss"],
                spatial_loss = loss["spatial_loss"],
                grad_loss=loss["grad_loss"],
                normal_loss=loss["normal_loss"],

                div_loss=loss["div_loss"],
                bce_loss=loss["bce_loss"]
                #inner_mse=collated_history["mse"],
                #inner_psnr=collated_history["psnr"],
            )
            accm.update(metrics, count=1)
            total_step += 1

            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)



        summary = accm.get_summary()
        summary["xs"] = xt
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch>=50 and epoch % self.config.experiment.test_imlog_freq == 0:
            #self.reconloggingstruct(summary["xs"], upsample_ratio=1, epoch=epoch, mode=mode)
            if self.config.dataset.type == 'shapenet':
                if self.config.type !='overfit':
                    model = self.model
                    model.eval()
                    xt = summary["xs"]
                    coords = xt['coords'].to(self.device)
                    coords.requires_grad_()
                    labels = xt['label'].to(self.device)

                    if self.config.dataset.supervision == 'sdf':
                        xs = xt['sdf'].to(self.device)
                        normals = xt['normal'].to(self.device)
                        xs = torch.concatenate([xs, normals], dim=-1)
                    else:
                        xs = xt['occ'].to(self.device)

                    vis = True
                    _, meshes, _ = model(xs, coords, is_training=False, vis=vis)
                    self.reconstruct_shape(meshes,epoch,mode=mode)
                else:
                    model = self.model
                    model.eval()
                    xs = summary["xs"]
                    coords = xs['coords'].to(self.device)
                    #xs = xs['occ'].to(self.device)

                    vis = True
                    meshes = model.overfit_one_shape(xs, coord=coords,vis=vis,type=self.config.dataset.supervision)
                    self.reconstruct_shape(meshes,epoch,mode=mode)

                #self.writer.add_mesh(mode=mode, tag='my_mesh', vertices=vertices_tensor, colors=colors_tensor,
                #                     faces=faces_tensor)

                #for i in range(len(meshes)):
                #    break
                #    vertices_tensor = torch.as_tensor(meshes[i]['vertices'].copy(), dtype=torch.int).unsqueeze(0)
                #    faces_tensor=torch.as_tensor(meshes[i]['faces'].copy(), dtype=torch.int).unsqueeze(0)
                #   color = np.array([[255, 154, 234]])
                #    color = np.repeat(color, meshes[i]['vertices'].shape[0], axis=0)
                #    colors_tensor = torch.as_tensor(color, dtype=torch.int).unsqueeze(0)
                #    self.writer.add_mesh(mode=mode,tag='my_mesh', vertices=vertices_tensor, colors=colors_tensor,faces=faces_tensor)

            else:
                self.reconstruct(summary["xs"], upsample_ratio=3, epoch=epoch, mode=mode)


        self.writer.add_scalar("loss/loss_total", summary["loss_total"], mode, epoch)
        self.writer.add_scalar("loss/onsurface_loss", summary["onsurface_loss"], mode, epoch)
        self.writer.add_scalar("loss/spatial_loss", summary["spatial_loss"], mode, epoch)
        self.writer.add_scalar("loss/grad_loss", summary["grad_loss"], mode, epoch)
        self.writer.add_scalar("loss/normal_loss", summary["normal_loss"], mode, epoch)
        self.writer.add_scalar("loss/div_loss", summary["div_loss"], mode, epoch)
        self.writer.add_scalar("loss/bce_loss", summary["bce_loss"], mode, epoch)

        #self.writer.add_scalar("loss/mse", summary["mse"], mode, epoch)
        #self.writer.add_scalar("loss/psnr", summary["psnr"], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)
            self.writer.add_scalar("inner_lr", self.model.get_lr(), mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, xs, upsample_ratio=1, epoch=0, mode="valid"):
        r"""Reconstruct the input data according to `upsample_ratio` and logs the results
        Args
            xs (torch.Tensor) : the data to be reconstructed.
            upsample_ratio (int, float) : upsamling ratio in (0, \inf) for data ireconstruction.
                If `upsample_ratio<1` the reconstructed results will be down-sampled.
                If `upsample_ratio==1`, the reconstructed data have the same resolution with the input data `xs`.
                If `upsample_ratio>1` the reconstructed results have higher resolution than input data using coordinate interpolation of INRs.
            epoch (int) : the number of epoch to be logged.
            mode (str) : the prefix for logging the result (e.g. "valid, "train")
        """


        def get_recon_imgs(xs_real, xs_recon, upsample_ratio=1):
            xs_real = xs_real
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(xs_real, scale_factor=upsample_ratio)
                xs_recon = torch.nn.functional.interpolate(xs_recon, scale_factor=upsample_ratio)
            xs_recon = torch.clamp(xs_recon, 0, 1)
            return xs_real, xs_recon

        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        assert upsample_ratio > 0

        xs_real = xs[:4].to(self.device)
        #coord_inputs = model.module.sample_coord_input(xs_real, upsample_ratio=upsample_ratio, device=xs.device)
        coord_inputs = model.sample_coord_input(xs_real, device=self.device)

        xs_recon, _, collated_history = model(xs_real, coord_inputs, is_training=False)

        xs_real, xs_recon = get_recon_imgs(xs_real, xs_recon, upsample_ratio)

        grid = []
        if upsample_ratio == 1:
            inner_step_recons = collated_history["recons"].clamp(0, 1)
            grid.append(inner_step_recons)
        grid.extend([xs_recon.unsqueeze(1), xs_real.unsqueeze(1)])

        grid = torch.cat(grid, dim=1)
        nrow = grid.shape[1]
        grid = grid.reshape(-1, *xs_recon.shape[1:])
        grid = torchvision.utils.make_grid(grid, nrow=nrow)
        self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
