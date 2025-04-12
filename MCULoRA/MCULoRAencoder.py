import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention_softmoe import *
from modules.Lora import MCULoRALinear

class MCULoRA(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(MCULoRA, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate


        # in_features = self.adim
        out_features = D_e
        # r = {'0': 5, '1':5, '2': 5, '3': 5, '4':5, '5': 5, '6': 5, '7': 5}
        r = {'0': 4, '1':4, '2': 4, '3': 4, '4':4, '5': 4, '6': 4, '7': 4}
        lora_shared_scale = 1.0
        lora_task_scale={'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0 ,'4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0}
        lora_dropout = 0.1
        tasks = ['0','1','2','3','4','5','6','7']
        trainable_scale_shared = False
        trainable_scale_per_task = False
        shared_mode = 'matrix'

    # 创建 MTLoRALinear 实例
        self.a_in_proj = MCULoRALinear(
        in_features=self.adim,
        out_features=out_features,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )
        self.t_in_proj = MCULoRALinear(
        in_features=self.tdim,
        out_features=out_features,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )
        self.v_in_proj = MCULoRALinear(
        in_features=self.vdim,
        out_features=out_features,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )

        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)



        self.block = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=4,
                )
        # self.proj1 = nn.Linear(D, D)


        self.proj1 = MCULoRALinear(
        in_features=D,
        out_features=D,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        # self.nlp_head = nn.Linear(D, n_classes)
        self.nlp_head = MCULoRALinear(
        in_features=D,
        out_features=n_classes,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )
        self.router_out = nn.Linear(3 * D_e, 1)
        self.router_out_lora = nn.Linear(3 * D_e, 1)

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False,index='0'):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])
        weight_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)

        proj_a,lora_as=self.a_in_proj(audio)
        # print(index)
        # proj_a=lora_as[index]+proj_a
        proj_t,lora_ts=self.t_in_proj(text)
        # proj_t=lora_ts[index]+proj_t
        # proj_t=lora_ts[str[index]]+proj_t
        proj_v,lora_vs=self.v_in_proj(video)
        # proj_v=lora_vs[index]+proj_v
        # proj_v=lora_vs[str[index]]+proj_v
        proj_a = self.dropout_a(proj_a)
        proj_t = self.dropout_t(proj_t)
        proj_v = self.dropout_v(proj_v)
       
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        # --> [batch, 3*seqlen, dim]
        x_a,xs_a = self.block(proj_a, True, attn_mask, 'a',xs=lora_as)
        x_t,xs_t = self.block(proj_t, True, attn_mask, 't',xs=lora_ts)
        x_v,xs_v = self.block(proj_v, True, attn_mask, 'v',xs=lora_vs)

        # if first_stage:
        out_a = self.nlp_head_a(x_a)
        out_t = self.nlp_head_t(x_t)
        out_v = self.nlp_head_v(x_v)

        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        res = x_joint
        
        # for a, t, v in zip(xs_a, xs_t, xs_v):
        #     xs_all = torch.cat([a, t, v], dim=-1)
        xs_all = {}
        for key in xs_a.keys():
            a = xs_a[key]
            t = xs_t[key]
            v = xs_v[key]
            xs_all[key] = torch.cat([a, t, v], dim=-1)
        x_all,xs_all=self.proj1(x_joint,xs_all)
        # res=x_all+xs_all[index]
        u = F.relu(x_all)
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out,out_lora = self.nlp_head(hidden,xs_all)
        out=out+out_lora[index]

        return hidden, out, out_a, out_t, out_v, np.array(weight_save),x_a,x_t,x_v,xs_a,xs_t,xs_v


if __name__ == '__main__':
    input = [torch.randn(61, 32, 300)]
    model = MCULoRA(100, 100, 100, 128, 1)
    anchor = torch.randn(32, 61, 128)
    hidden, out, _ = model(input)
