import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import torch.nn.functional as F

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_ray import render_rays
from gnt.render_image import render_single_image
from gnt.model import GNTModel, GNTMoEModel
from gnt.sample_ray import RaySamplerSingleImage
from gnt.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr,collect_noisy_gating_loss, distance
import config
import torch.distributed as dist
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset
import imageio

from torch.utils.tensorboard import SummaryWriter   

import fmoe
import datetime

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # resfile = open('distance.txt','w')

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # log file
    if args.local_rank==0:
        writer = SummaryWriter(out_folder)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create GNT model
    moe_save_flag = args.arch.endswith('moe') and (not args.moe_data_distributed)
    if args.arch.endswith('gnt'):
        model = GNTModel(
            args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
        )
    elif args.arch.endswith('moe'):
        model = GNTMoEModel(
            args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
        )
    if args.local_rank==0:
        state = (model.net_coarse).state_dict()
        for key, item in state.items():
            print(key, item.shape)
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop
            
            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device, consistency=args.consistency)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            )
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )

            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2)) #roughly H/4 W/4, has some paddings
            
            target_featmaps = None
            if args.use_target:
                target_featmaps = model.feature_net(train_data["rgb"].permute(0, 3, 1, 2).to(device))
                for i in range(len(target_featmaps)):
                    temp = target_featmaps[i]
                    temp = F.interpolate(temp, train_data["rgb"].shape[1])
                    temp = temp.reshape(-1, target_featmaps[i].shape[1])[ray_batch["selected_inds"]]
                    target_featmaps[i] = torch.cat([ray_batch["rgb"], temp], dim=-1)
            
            ret = render_rays(
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                N_importance=args.N_importance,
                det=args.det,
                white_bkgd=args.white_bkgd,
                ret_alpha=args.N_importance > 0,
                single_net=args.single_net,
                consistency=args.consistency,
                target_featmaps=target_featmaps,
            )

            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log)
            
            if args.arch.endswith('gntmoe'):
                gating_loss = collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                loss += gating_loss
            if args.consistency:
                EPSILON = 10e-3
                Delta = 10e3
                close_rays = ray_batch["close_inds"]
                logits = torch.stack(ret["outputs_coarse"]["moe_logits"])
                logits = logits.view(logits.shape[0], N_rand, -1, 4)
                pts = ret["outputs_coarse"]["pts"]
                ids=np.nonzero(close_rays)[0]
                rays_a = torch.zeros([logits.shape[0], len(ids), logits.shape[2], 4])
                rays_b = torch.zeros([logits.shape[0], len(ids), logits.shape[2], 4])
                dist_all = torch.zeros([len(ids), logits.shape[2]])
                rays_b_s=[]
                for ii, idx in enumerate(ids):
                    rays_a[:,ii:ii+1,:,:] = logits[:,idx:idx+1,:,:]
                    rays_b[:,ii:ii+1,:,:] = logits[:,idx-1:idx,:,:]
                    dist = distance(pts[idx,:,:], pts[idx-1,:,:])
                    val, ind = dist.min(dim=-1)
                    dist_all[ii] = val
                    rays_b_s.append(rays_b[:,ii:ii+1,ind,:])
                if len(rays_b_s)>0:
                    rays_b_sel= torch.stack(rays_b_s, dim=1).squeeze(dim=2)
                    kl = F.kl_div(rays_a.log(), rays_b_sel, reduction='none')
                    kl = torch.mean(kl, dim=(0,3), keepdim=True).squeeze()
                    dist_all[dist_all != dist_all]=Delta
                    dist_all = torch.clamp(dist_all, min=EPSILON, max=Delta)
                    weights = F.softmax(-dist_all, dim=1)
                    consistency_loss = torch.mean(kl*weights).to(device=loss.device)*1e5
                    loss += consistency_loss

            if ret["outputs_fine"] is not None:
                fine_loss, scalars_to_log = criterion(
                    ret["outputs_fine"], ray_batch, scalars_to_log
                )
                loss += fine_loss

            loss.backward()
            scalars_to_log["loss"] = loss.item()
            if args.arch.endswith('moe'):
                model.net_coarse.allreduce_params()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            rank = args.local_rank
            if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
                # write mse and psnr stats
                mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                scalars_to_log["coarse-loss"] = mse_error
                psnr = mse2psnr(mse_error)
                
                writer.add_scalar("coarse_loss", mse_error, global_step)
                writer.add_scalar('gating_loss', gating_loss, global_step)
                scalars_to_log["gating_loss"] = gating_loss
                scalars_to_log["coarse-psnr-training-batch"] = psnr
                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('psnr_train', psnr, global_step)
                if ret["outputs_fine"] is not None:
                    mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                    scalars_to_log["train/fine-loss"] = mse_error
                    scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)
            
                logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                for k in scalars_to_log.keys():
                    logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                print(logstr)
                print("each iter time {:.05f} seconds".format(dt))

            if global_step % args.i_weights == 0:
                print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                model.save_model(fpath, moe_save_flag)

            if rank == 0 and global_step % args.i_img == 0:
                print("Logging a random validation view...")
                val_data = next(val_loader_iterator)
                tmp_ray_sampler = RaySamplerSingleImage(
                    val_data, device, render_stride=args.render_stride
                )
                H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                log_view(
                    global_step,
                    args,
                    model,
                    tmp_ray_sampler,
                    projector,
                    gt_img,
                    render_stride=args.render_stride,
                    prefix="val/",
                    out_folder=out_folder,
                    ret_alpha=args.N_importance > 0,
                    single_net=args.single_net,
                )
                torch.cuda.empty_cache()

                print("Logging current training view...")
                tmp_ray_train_sampler = RaySamplerSingleImage(
                    train_data, device, render_stride=1
                )
                H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                log_view(
                    global_step,
                    args,
                    model,
                    tmp_ray_train_sampler,
                    projector,
                    gt_img,
                    render_stride=1,
                    prefix="train/",
                    out_folder=out_folder,
                    ret_alpha=args.N_importance > 0,
                    single_net=args.single_net,
                )
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2)) #roughly H/4 W/4, has some paddings
            target_featmaps = None
            if args.use_target:
                target_img = gt_img[None, :, :, :]
                target_featmaps = model.feature_net(target_img.permute(0, 3, 1, 2).to(featmaps[0].device))
                for i in range(len(target_featmaps)):
                    temp = target_featmaps[i]
                    temp = F.interpolate(temp, target_img.shape[1])
                    temp = temp.reshape(-1, target_featmaps[i].shape[1])
                    target_featmaps[i] = torch.cat([ray_batch["rgb"], temp], dim=-1)
        else:
            featmaps = [None, None]
            target_featmaps = None
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=ret_alpha,
            single_net=single_net,
            target_featmaps=target_featmaps,
        )

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    model.switch_to_train()


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400))
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)
