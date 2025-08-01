import torch
from torch import nn
import torch.nn.functional as F
from .Lora import MCULoRALinear

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

         # in_features = self.adim
        # out_features = D_e
        # r = {'0': 5, '1':5, '2': 5, '3': 5, '4':5, '5': 5, '6': 5, '7': 5}
        r = {'0': 8, '1':8, '2': 8, '3': 8, '4':8, '5': 8, '6': 8, '7': 8}
        # r = {'0': 4, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4}
        lora_shared_scale = 1.0
        lora_task_scale={'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0 ,'4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0}
        lora_dropout = 0.1
        tasks = ['0','1','2','3','4','5','6','7']
        trainable_scale_shared = False
        trainable_scale_per_task = False
        shared_mode = 'matrix'

    # 创建 MTLoRALinear 实例
        self.fc1 = MCULoRALinear(
        in_features=in_features,
        out_features=hidden_features,
        r=r,
        lora_shared_scale=lora_shared_scale,
        lora_task_scale=lora_task_scale,
        lora_dropout=lora_dropout,
        tasks=tasks,
        trainable_scale_shared=trainable_scale_shared,
        trainable_scale_per_task=trainable_scale_per_task,
        shared_mode=shared_mode
        )

        self.fc2 = MCULoRALinear(
        in_features=hidden_features,
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



        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,xs=None):
        x,xs = self.fc1(x,xs)
        x = self.act(x)
        x = self.drop(x)
        x,xs = self.fc2(x,xs)
        x = self.drop(x)
        return x,xs


class Attention_group(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Transformer_a = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, cross_modality='atv', mask_modality=None,xs=None, mask=None):
        # x: [B, s, C]
        B, s, C = x.shape
        # 仅仅单模态特性信息的自注意力
        if cross_modality == 'a':
            x_a_mlp,xs = self.Transformer_a(x, mask_modality,xs, mask)
            return x_a_mlp,xs
        if cross_modality == 't':
            x_t_mlp,xs = self.Transformer_t(x, mask_modality,xs, mask)
            return x_t_mlp,xs
        if cross_modality == 'v':
            x_v_mlp,xs = self.Transformer_v(x, mask_modality,xs, mask)
            return x_v_mlp,xs



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )


    def forward(self, x, mask_modality,xs=None, mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        if mask is not None:
            mask = mask.bool()
            mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
            mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_lora,xs=self.mlp(x_out,xs)
        x_out = x_out + x_lora
        
        return x_out,xs


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.drop = drop
        # 根据模态组合任务，选择不同的block
        self.blocks = nn.ModuleList(
            [
                Attention_group(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio,)
                for i in range(depth)
            ]
        )

    def forward(self, x, first_stage, mask=None, modality=None,xs=None):
        # 根据模态组合任务，选择不同的block
        for layer_idx, block in enumerate(self.blocks):
                x_res,xs=block(x, cross_modality=modality, mask_modality=modality, mask=mask,xs=xs)
                x = x + x_res
        return x,xs
