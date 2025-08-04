import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention_group import *
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
        
        # 跨任务注意力机制
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=D_e,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            batch_first=True
        )
        
        # 跨模态注意力机制
        self.cross_modality_attention_at = nn.MultiheadAttention(  # audio-text
            embed_dim=D_e,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            batch_first=True
        )
        self.cross_modality_attention_av = nn.MultiheadAttention(  # audio-visual
            embed_dim=D_e,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            batch_first=True
        )
        self.cross_modality_attention_tv = nn.MultiheadAttention(  # text-visual
            embed_dim=D_e,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            batch_first=True
        )
    
    def apply_cross_task_attention(self, xs_dict):
        """
        对xs_dict中每个key的特征进行跨任务注意力机制
        xs_dict: 字典，包含不同任务的特征 {'0': tensor, '1': tensor, ...}
        """
        if xs_dict is None or len(xs_dict) == 0:
            return xs_dict
            
        # 获取所有任务的特征
        task_keys = list(xs_dict.keys())
        task_features = []
        
        for key in task_keys:
            task_features.append(xs_dict[key])
        
        # 将所有特征堆叠成 [num_tasks, batch, seq_len, dim]
        stacked_features = torch.stack(task_features, dim=0)  # [num_tasks, batch, seq_len, dim]
        num_tasks, batch_size, seq_len, dim = stacked_features.shape
        
        # 重塑为 [num_tasks, batch*seq_len, dim] 用于注意力计算
        reshaped_features = stacked_features.view(num_tasks, batch_size * seq_len, dim)
        
        # 应用跨任务注意力
        attended_features, _ = self.cross_task_attention(
            query=reshaped_features,
            key=reshaped_features,
            value=reshaped_features
        )
        
        # 重塑回原始形状
        attended_features = attended_features.view(num_tasks, batch_size, seq_len, dim)
        
        # 更新字典
        updated_xs_dict = {}
        for i, key in enumerate(task_keys):
            updated_xs_dict[key] = attended_features[i]
        
        return updated_xs_dict
    
    def apply_cross_modality_attention(self, xs_a, xs_t, xs_v):
        """
        对xs_a、xs_t、xs_v进行跨模态相互注意力
        实现不同模态之间的相互注意力机制
        """
        if xs_a is None or xs_t is None or xs_v is None:
            return xs_a, xs_t, xs_v
            
        # 获取所有任务的特征
        task_keys = list(xs_a.keys())
        
        # 初始化输出字典
        updated_xs_a = {}
        updated_xs_t = {}
        updated_xs_v = {}
        
        for key in task_keys:
            # 获取当前任务的三个模态特征
            a_feat = xs_a[key]  # [batch, seq_len, dim]
            t_feat = xs_t[key]  # [batch, seq_len, dim]
            v_feat = xs_v[key]  # [batch, seq_len, dim]
            
            # 1. Audio-Text 跨模态注意力
            # Query: text, Key&Value: audio
            at_attended_t, _ = self.cross_modality_attention_at(
                query=t_feat,
                key=a_feat,
                value=a_feat
            )
            
            # Query: audio, Key&Value: text
            ta_attended_a, _ = self.cross_modality_attention_at(
                query=a_feat,
                key=t_feat,
                value=t_feat
            )
            
            # 2. Audio-Visual 跨模态注意力
            # Query: visual, Key&Value: audio
            av_attended_v, _ = self.cross_modality_attention_av(
                query=v_feat,
                key=a_feat,
                value=a_feat
            )
            
            # Query: audio, Key&Value: visual
            va_attended_a, _ = self.cross_modality_attention_av(
                query=a_feat,
                key=v_feat,
                value=v_feat
            )
            
            # 3. Text-Visual 跨模态注意力
            # Query: visual, Key&Value: text
            tv_attended_v, _ = self.cross_modality_attention_tv(
                query=v_feat,
                key=t_feat,
                value=t_feat
            )
            
            # Query: text, Key&Value: visual
            vt_attended_t, _ = self.cross_modality_attention_tv(
                query=t_feat,
                key=v_feat,
                value=v_feat
            )
            
            # 融合所有跨模态注意力结果
            # 音频特征融合：原始 + 从文本获取的信息 + 从视觉获取的信息
            updated_a = a_feat + ta_attended_a + va_attended_a
            
            # 文本特征融合：原始 + 从音频获取的信息 + 从视觉获取的信息
            updated_t = t_feat + at_attended_t + vt_attended_t
            
            # 视觉特征融合：原始 + 从音频获取的信息 + 从文本获取的信息
            updated_v = v_feat + av_attended_v + tv_attended_v
            
            # 保存更新后的特征
            updated_xs_a[key] = updated_a
            updated_xs_t[key] = updated_t
            updated_xs_v[key] = updated_v
        
        return updated_xs_a, updated_xs_t, updated_xs_v

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
        # sequence modeling,获得序列化表征
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)

        proj_a,lora_as=self.a_in_proj(audio)
  
        proj_t,lora_ts=self.t_in_proj(text)

        proj_v,lora_vs=self.v_in_proj(video)
 
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
        
        # 对xs_a、xs_t、xs_v进行跨模态相互注意力
        xs_a, xs_t, xs_v = self.apply_cross_modality_attention(xs_a, xs_t, xs_v)
        

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
        # 这里是联合梯度
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
