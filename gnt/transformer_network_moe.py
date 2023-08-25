import numpy as np
import torch
import torch.nn as nn
from .custom_moe_layer import FMoETransformerMLP, FMoETransformerMLP_Attention
from gnt.gate_funs.noisy_gate import NoisyGate
from gnt.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE 

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate, attention_moe=False, activation=nn.GELU, 
                num_expert=4, d_model=64, d_gate=64, d_hidden=64,
                world_size=1, top_k=2, gate=NoisyGate_VMoE,
                gate_return_decoupled_activation=False, vmoe_noisy_std=1):
        super(Attention2D, self).__init__()
        if attention_moe:
            self.q_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
            self.k_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
            self.v_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
        else:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
            self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate,
                act_layer=nn.GELU, attention_moe=False, moe=False, moe_mlp_ratio=-1, moe_experts=64,
                moe_top_k=2, moe_gate_dim=-1, world_size=1, gate_return_decoupled_activation=False,
                moe_gate_type="noisy", vmoe_noisy_std=1,gate_input_ahead = False, exp_force=False, 
                consistency=False, add_fix_exp=False, fix_mlp_ratio=4):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.moe = moe
        self.consistency = consistency
        self.add_fix_exp = add_fix_exp

        if moe or attention_moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(ff_dp_rate)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            if moe_gate_type == "noisy":
                moe_gate_fun = NoisyGate
            elif moe_gate_type == "noisy_vmoe":
                moe_gate_fun = NoisyGate_VMoE
            else:
                raise ValueError("unknow gate type of {}".format(moe_gate_type))


        if attention_moe:
            self.attn = Attention2D(dim, attn_dp_rate, attention_moe=True, num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                    world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                    gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
        else:
            self.attn = Attention2D(dim, attn_dp_rate, attention_moe=False)
        self.gate_input_ahead = gate_input_ahead
            
        
        if self.moe:
            self.mlp = FMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std, exp_force=exp_force,
                                          consistency=consistency)
            self.mlp_drop = nn.Dropout(ff_dp_rate)
            if self.add_fix_exp:
                fix_mlp_hidden_dim = int(dim * fix_mlp_ratio)
                self.fix_mlp = FeedForward(dim, fix_mlp_hidden_dim, ff_dp_rate)
        else:
            self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        

    def forward(self, q, k, pos, mask=None, gate_inp=None, dataset_id=None):
        if self.gate_input_ahead:
            gate_inp = q

        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        if self.moe:
            residue = x
            x1 = x2 = self.ff_norm(x)
            if self.consistency:
                x1, logits = self.mlp(x1, gate_inp, dataset_id=dataset_id)
            else:
                x1 = self.mlp(x1, gate_inp, dataset_id=dataset_id)
            x = self.mlp_drop(x1)
            if self.add_fix_exp:
                x2 = self.fix_mlp(x2)
                x = (x + x2)/2
            x = x + residue
        else:
            residue = x
            x = self.ff_norm(x)
            x = self.ff(x)
            x = x + residue

        if self.consistency and self.moe:
            return x, logits
        else:
            return x, None


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None,
                activation=nn.GELU, attention_moe=False,
                num_expert=4, d_model=64, d_gate=64, d_hidden=64,
                world_size=1, top_k=2, gate=NoisyGate_VMoE,
                gate_return_decoupled_activation=False, vmoe_noisy_std=1):
        super(Attention, self).__init__()
        if attention_moe:
            if attn_mode in ["qk", "gate"]:
                self.q_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
                self.k_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
                self.v_fc = FMoETransformerMLP_Attention(num_expert=num_expert, d_model=d_model, d_gate=d_gate, d_hidden=d_hidden,
                                          world_size=world_size, top_k=top_k, activation=activation, gate=gate,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
        else:
            if attn_mode in ["qk", "gate"]:
                self.q_fc = nn.Linear(dim, dim, bias=False)
                self.k_fc = nn.Linear(dim, dim, bias=False)
            if attn_mode in ["pos", "gate"]:
                self.pos_fc = nn.Sequential(
                    nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
                )
                self.head_fc = nn.Linear(dim // 8, n_heads)
            if attn_mode == "gate":
                self.gate = nn.Parameter(torch.ones(n_heads))
            self.v_fc = nn.Linear(dim, dim, bias=False)

        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None,
        act_layer=nn.GELU, attention_moe=False, moe=False, moe_mlp_ratio=-1, moe_experts=64,
        moe_top_k=2, moe_gate_dim=-1, world_size=1, gate_return_decoupled_activation=False,
        moe_gate_type="noisy", vmoe_noisy_std=1, gate_input_ahead = False, exp_force=False, 
        consistency=False, add_fix_exp=False, fix_mlp_ratio=4):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.moe=moe
        self.consistency = consistency
        self.add_fix_exp = add_fix_exp

        if moe or attention_moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(ff_dp_rate)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            if moe_gate_type == "noisy":
                moe_gate_fun = NoisyGate
            elif moe_gate_type == "noisy_vmoe":
                moe_gate_fun = NoisyGate_VMoE
            else:
                raise ValueError("unknow gate type of {}".format(moe_gate_type))
        if attention_moe:
            self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, attention_moe=True, 
                                num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std)
                                    
        else:
            self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, attention_moe=False)
        self.gate_input_ahead = gate_input_ahead


        if self.moe:
            self.mlp = FMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std, 
                                          exp_force=exp_force, consistency=consistency)
            self.mlp_drop = nn.Dropout(ff_dp_rate)
            if self.add_fix_exp:
                fix_mlp_hidden_dim = int(dim * fix_mlp_ratio)
                self.fix_mlp = FeedForward(dim, fix_mlp_hidden_dim, ff_dp_rate)
        else:
            self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)

    def forward(self, x, pos=None, ret_attn=False, gate_inp=None, dataset_id=None):
        if self.gate_input_ahead:
            gate_inp = x

        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, pos, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        if self.moe:
            residue = x
            x1 = x2 = self.ff_norm(x)
            if self.consistency:
                x1, logits = self.mlp(x1, gate_inp, dataset_id=dataset_id)
            else:
                x1 = self.mlp(x1, gate_inp, dataset_id=dataset_id)
            x = self.mlp_drop(x1)
            if self.add_fix_exp:
                x2 = self.fix_mlp(x2)
                x = (x + x2)/2
            x = x + residue
        else:
            residue = x
            x = self.ff_norm(x)
            x = self.ff(x)
            x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        if self.consistency and self.moe:
            return x, logits
        else:
            return x, None


class GNTMoE(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False,
                moe_mlp_ratio=-1, moe_experts=64, moe_top_k=2, world_size=1, gate_dim=-1,
                gate_return_decoupled_activation=False, moe_gate_type="noisy", vmoe_noisy_std=1, gate_input_ahead = False):
        super(GNTMoE, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        moe_pos = args.moe_pos
        moe_type = args.moe_type
        self.exp_force = args.exp_force
        attention_moe = args.moe_type=='att'

        self.view_trans = nn.ModuleList([])
        self.ray_trans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        #moe
        self.moe_experts = moe_experts
        self.moe_top_k = moe_top_k
        self.gate_return_decoupled_activation = gate_return_decoupled_activation
        self.gate_input_ahead = gate_input_ahead
        self.consistency = args.consistency
        for i in range(args.trans_depth):
            if i % 2 == 0:
                # view transformer
                view_trans = Transformer2D(
                    dim=args.netwidth,
                    ff_hid_dim=int(args.netwidth * 4),
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                )
                self.view_trans.append(view_trans)
                # ray transformer
                ray_trans = Transformer(
                    dim=args.netwidth,
                    ff_hid_dim=int(args.netwidth * 4),
                    n_heads=4,
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                    attn_mode="qk"
                )
                self.ray_trans.append(ray_trans)
                # mlp
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
                self.q_fcs.append(q_fc)
            else:
                view_trans = Transformer2D(
                    dim=args.netwidth,
                    ff_hid_dim=int(args.netwidth * 4),
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                    attention_moe = attention_moe,
                    moe=not attention_moe, moe_mlp_ratio=moe_mlp_ratio, moe_experts=moe_experts, moe_top_k=moe_top_k, moe_gate_dim=gate_dim, world_size=world_size,
                    gate_return_decoupled_activation=self.gate_return_decoupled_activation,
                    moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std,gate_input_ahead=self.gate_input_ahead, exp_force = self.exp_force, consistency=args.consistency,
                    add_fix_exp=args.add_fix_exp, fix_mlp_ratio=args.fix_mlp_ratio,
                )
                self.view_trans.append(view_trans)
                # ray transformer
                ray_trans = Transformer(
                    dim=args.netwidth,
                    ff_hid_dim=int(args.netwidth * 4),
                    n_heads=4,
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                    attn_mode="qk"
                )
                self.ray_trans.append(ray_trans)
                # mlp
                q_fc = nn.Identity()
                self.q_fcs.append(q_fc)
            

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d, gate_inp=None, dataset_id=None):
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]

        # transformer modules
        moe_logits=[]
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_trans, self.q_fcs, self.ray_trans)
        ):
            if i % 2 ==0:
                # view transformer to update q
                q,_ = crosstrans(q, rgb_feat, ray_diff, mask)
                # embed positional information
                q = torch.cat((q, input_pts, input_views), dim=-1)
                q = q_fc(q)
                # ray transformer
                if self.ret_alpha:
                    q = selftrans(q, ret_attn=self.ret_alpha)
                    q, attn = q
                else:
                    q,_ = selftrans(q, ret_attn=self.ret_alpha)
            else:
                # view transformer to update q
                q, logits_view = crosstrans(q, rgb_feat, ray_diff, mask, gate_inp=gate_inp, dataset_id=dataset_id)
                if logits_view!=None:
                    moe_logits.append(logits_view )
                # ray transformer
                q, logits_ray = selftrans(q, ret_attn=self.ret_alpha, gate_inp=gate_inp, dataset_id=dataset_id)
                if logits_ray!=None:
                    moe_logits.append(logits_ray )
            
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        if self.consistency:
            return outputs, moe_logits
        else:
            return outputs
