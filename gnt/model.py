import torch
import os
from gnt.transformer_network import GNT
from gnt.transformer_network_moe import GNTMoE
from gnt.feature_network import ResUNet

import fmoe
from utils import sync_weights

import torch.nn as nn

from collections import OrderedDict

def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        # create coarse GNT
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        ).to(device)
        # single_net - trains single network which can be used for both coarse and fine sampling
        if args.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }

        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        # print(to_load["net_coarse"].keys())
        # import pdb; pdb.set_trace()
        # exit()
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"])

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step


class GNTMoEModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        if self.args.moe_data_distributed:
            self.moe_world_size = 1
        else:
            self.moe_world_size = torch.distributed.get_world_size()
            if self.args.moe_experts % self.moe_world_size != 0:
                print("experts number of {} is not divisible by world size of {}".format(args.moe_experts, self.moe_world_size))
            self.args.moe_experts = self.args.moe_experts // self.moe_world_size
        self.net_coarse = GNTMoE(
                        args,
                        in_feat_ch=self.args.coarse_feat_dim,
                        posenc_dim=3 + 3 * 2 * 10,
                        viewenc_dim=3 + 3 * 2 * 10,
                        ret_alpha=self.args.N_importance > 0,
                        moe_experts=self.args.moe_experts, moe_top_k=self.args.moe_top_k,
                        world_size=self.moe_world_size,
                        moe_mlp_ratio=self.args.moe_mlp_ratio,
                        moe_gate_type=self.args.moe_gate_type, vmoe_noisy_std=self.args.vmoe_noisy_std,
                        gate_input_ahead = self.args.gate_input_ahead
                    ).to(device)

        if args.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNTMoE(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
                moe_experts=self.args.moe_experts, moe_top_k=self.args.moe_top_k,
                world_size=self.moe_world_size,
                moe_mlp_ratio=self.args.moe_mlp_ratio,
                moe_gate_type=self.args.moe_gate_type, vmoe_noisy_std=self.args.vmoe_noisy_std,
                gate_input_ahead = self.args.gate_input_ahead
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.net_fine.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            if "moe" in args.arch:
                self.net_coarse = fmoe.DistributedGroupedDataParallel(self.net_coarse, device_ids=[args.local_rank])
                sync_weights(self.net_coarse, except_key_words=["experts.h4toh", "experts.htoh4"])

            else:
                self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                    self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
                )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def filter_state(self, state):
        from collections import OrderedDict
        new_state = OrderedDict()
        for key, item in state.items():
            if "experts.htoh4" in key or "experts.h4toh" in key:
                new_state[key] = item
        return new_state

    def save_moe_model(self, state, filename_temp):
        rank = torch.distributed.get_rank()     

        save_name = filename_temp.replace('.pth', "_{}.pth".format(rank))
        if rank != 0:
            state["net_coarse"] = self.filter_state(state["net_coarse"])
            torch.save(state["net_coarse"], save_name)
        else:
            torch.save(state, save_name)

    def save_model(self, filename, moe_save=True):
        to_save = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net_coarse": de_parallel(self.net_coarse).state_dict(),
                "feature_net": de_parallel(self.feature_net).state_dict(),
                }
        if self.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.net_fine).state_dict()

        if moe_save:
            filename_temp = filename
            self.save_moe_model(to_save, filename_temp)
        else:
            torch.save(to_save,filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])

        if self.net_fine is not None and "net_fine" in to_load.keys():
            self.net_fine.load_state_dict(to_load["net_fine"])

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith("0.pth")
            ]
        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                if self.args.ckpt_path.endswith('_0.pth'):
                    ckpts = [self.args.ckpt_path]
                else:
                    ckpts = [self.args.ckpt_path]
                    fpath = self.args.ckpt_path
                    checkpoint = torch.load(fpath, map_location="cpu")
                    state =(self.net_coarse).state_dict()
                    new_state_dict = OrderedDict()
                    for key, item in state.items():
                        if "gate" in key:
                            gate_ini = torch.ones(item.shape[0],self.args.moe_experts)
                            new_state_dict[key]=gate_ini
                        elif "mlp.experts.htoh4.weight" in key:
                            new_state_dict[key]=checkpoint["net_coarse"][key.replace("mlp.experts.htoh4.weight", "ff.fc1.weight")].repeat(self.args.moe_experts,1,1)
                        elif "mlp.experts.htoh4.bias" in key:
                            new_state_dict[key]=checkpoint["net_coarse"][key.replace("mlp.experts.htoh4.bias", "ff.fc1.bias")].repeat(self.args.moe_experts,1)
                        elif "mlp.experts.h4toh.weight" in key:
                            new_state_dict[key]=checkpoint["net_coarse"][key.replace("mlp.experts.h4toh.weight", "ff.fc2.weight")].repeat(self.args.moe_experts,1,1)
                        elif "mlp.experts.h4toh.bias" in key:
                            new_state_dict[key]=checkpoint["net_coarse"][key.replace("mlp.experts.h4toh.bias", "ff.fc2.bias")].repeat(self.args.moe_experts,1)
                        else:
                            new_state_dict[key]=checkpoint["net_coarse"][key]

                    msg = self.net_coarse.load_state_dict(new_state_dict)
                    self.feature_net.load_state_dict(checkpoint["feature_net"])
                    if self.net_fine is not None and "net_fine" in to_load.keys():
                        self.net_fine.load_state_dict(to_load["net_fine"])
                    print('=================model unmatched keys:================',msg)
                    step = int(fpath[-10:-4])
                    print("Reloading from {} with repeated weights for MoE, starting at step={}".format(fpath, step))
                    return step
        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            checkpoint = torch.load(fpath, map_location="cpu")
            len_save = len([f for f in os.listdir(out_folder) if ".pth" in f]) // len(ckpts)

            assert len_save % self.moe_world_size == 0
            if self.args.distributed:
                response_cnt = [i for i in range(
                    torch.distributed.get_rank() * (len_save // self.moe_world_size),
                    (torch.distributed.get_rank() + 1) * (len_save // self.moe_world_size))]
            else:
                response_cnt = [i for i in range(0, len_save // self.moe_world_size)]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(fpath.replace("0.pth","{}.pth".format(cnt_model)),map_location="cpu")

                    if cnt != 0:
                        for key, item in checkpoint_specific.items():
                            checkpoint["net_coarse"][key] = torch.cat([checkpoint["net_coarse"][key], item],
                                                                        dim=0)
                    else:
                        checkpoint["net_coarse"].update(checkpoint_specific)
                    
            msg = self.net_coarse.load_state_dict(checkpoint['net_coarse'])
            self.feature_net.load_state_dict(checkpoint["feature_net"])
            if load_opt:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if load_scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if self.net_fine is not None and "net_fine" in checkpoint.keys():
                self.net_fine.load_state_dict(checkpoint["net_fine"])
            print('=================model unmatched keys:================',msg)
            step = int(fpath[-12:-6])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0
        return step