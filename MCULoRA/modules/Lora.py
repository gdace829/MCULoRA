import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple ,Mapping, Dict

class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

class MCULoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[int, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = 'matrix',
        **kwargs,
    ):
        # assert shared_mode in ['matrix', 'matrixv2',
        #                        'add', 'addition', 'lora_only']
       

        # if isinstance(r, int):
        #     r = {'shared': r}
        super().__init__(
            r=r['0'], lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)

        self.linear = torch.nn.Linear(
            in_features, out_features)
        # r = {'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2}
        self.tasks = tasks
        self.shared_mode = shared_mode
        self.lora_shared_scale=lora_shared_scale
        self.lora_task_scale=lora_task_scale
        self.r=r
        if self.r['0'] > 0:
           # 根据tasks的数量创建lora A矩阵的数
            self.lora_tasks_A = nn.ParameterDict({
                str(task): nn.Parameter(self.linear.weight.new_zeros((r[task], in_features)))
                for task in tasks
            })
            self.lora_tasks_B = nn.ParameterDict({
                str(task): nn.Parameter(self.linear.weight.new_zeros((out_features, r[task])))
                for task in tasks
            })
            # self.lora_task_scale={'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2, '6': 2, '7': 2}
            # if trainable_scale_per_task:
            #     self.lora_task_scale = nn.ParameterDict({
            #         str(task): nn.Parameter(torch.tensor([lora_task_scale[idx]]))
            #         for idx, task in enumerate(tasks)
            #     })
            self.lora_shared_A = nn.Parameter(
                    self.linear.weight.new_zeros((self.r['0'], in_features)))
            self.lora_shared_B = nn.Parameter(
                    self.linear.weight.new_zeros((out_features, self.r['0'])))
          
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(
                    torch.FloatTensor([lora_shared_scale]))
            else:
                self.lora_shared_scale = lora_shared_scale
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_tasks_A"):
            for task in self.tasks:
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[task])

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, x_tasks: Dict[int, torch.Tensor] = None,index:str=None):
        pretrained = self.linear(x)
        if self.r['0'] == 0:
            return pretrained, None
        x = self.lora_dropout(x)
            # 共享lora
        lora = (x @ self.lora_shared_A.transpose(0, 1)
                    @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale
            # 特定任务lora
        lora_tasks = {
            idx: pretrained + ((x if x_tasks is None else x_tasks[idx]) @ self.lora_tasks_A[idx].transpose(
                0, 1) @ self.lora_tasks_B[idx].transpose(0, 1) * self.lora_task_scale[idx])
            for idx in self.tasks
            } if self.tasks is not None else None
        
        # print(lora_tasks.keys())
        return pretrained+lora, lora_tasks



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
if __name__ == "__main__":
    # 设置参数
    in_features = 10
    out_features = 20
    r = {'0': 2, '1': 2, '2': 2}
    lora_shared_scale = 1.0
    lora_task_scale = [1.0,2.0,3.0]
    lora_dropout = 0.1
    tasks = ['0','1','2']
    trainable_scale_shared = False
    trainable_scale_per_task = False
    shared_mode = 'matrix'

    # 创建 MTLoRALinear 实例
    mtlora_linear = MTLoRALinear(
        in_features=in_features,
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

    # 准备输入数据
    batch_size = 32
    len=16
    x = torch.randn(batch_size,len, in_features)
    x_tasks = {
        '0': torch.randn(batch_size,len, in_features),
        '1': torch.randn(batch_size,len, in_features),
        '2': torch.randn(batch_size,len, in_features)
    }

    # 进行前向传播
    output, lora_tasks_output = mtlora_linear(x, None)

    # 打印输出结果
    print("共享 LoRA 调整后的输出形状:", output.shape)
    if lora_tasks_output is not None:
        for task, task_output in lora_tasks_output.items():
            print(f"任务 {task} 的 LoRA 调整后的输出形状:", task_output.shape)
    else:
        print("没有任务特定的 LoRA 调整输出。")
    