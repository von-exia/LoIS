import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from CLIP import clip



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
        

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



class Image_Encoder_V2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
        
    def forward(self, img):

        x = self.visual.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        

        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :])
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj
                
        return x, None
    
    

import torch
import torch.nn.functional as F
def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算给定logits中对应标签的对数概率。

    参数:
        logits (torch.Tensor): 模型输出的logits，形状为(batch_size, sequence_length, vocab_size)
        labels (torch.Tensor): 目标标签，形状为(batch_size, sequence_length)

    返回:
        torch.Tensor: 每个位置对应标签的对数概率，形状与labels相同
    """
    # 对logits进行log_softmax操作，得到对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 使用gather函数提取对应标签的log_probs
    # 将labels的维度从(batch_size, seq_len)扩展为(batch_size, seq_len, 1)
    labels = labels.unsqueeze(-1).long()
    
    # 在最后一个维度上收集对应的log_probs，结果的形状为(batch_size, seq_len, 1)
    selected_log_probs = log_probs.gather(dim=-1, index=labels)
    
    # 去除最后一个维度，恢复为(batch_size, seq_len)
    return selected_log_probs.squeeze(-1)


# def probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     log_probs = F.softmax(logits, dim=-1)
#     labels = labels.unsqueeze(-1).long()
#     selected_log_probs = log_probs.gather(dim=-1, index=labels)
#     return selected_log_probs.squeeze(-1)


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount=1):
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]

def discount_cumsum_batched(rewards, discount=1.0):
    """
    Compute discounted cumulative sums of vectors in batch mode.

    Args:
        rewards (torch.Tensor): A batch of reward sequences.
            Shape: (batch_size, sequence_length)
        discount (float): Discount factor (0 <= discount <= 1)

    Returns:
        torch.Tensor: Discounted cumulative sums. Same shape as input.
    """
    # 处理 discount=0 的特殊情况
    if discount == 0:
        return rewards.clone()
    
    # 处理 discount=1 的特殊情况（常规累加和）
    if discount == 1.0:
        return torch.cumsum(rewards.flip(dims=[1]), dim=1).flip(dims=[1])
    
    batch_size, seq_len = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    
    # 创建指数衰减系数矩阵
    powers = torch.arange(seq_len, device=device, dtype=dtype)
    exponent = powers - powers.unsqueeze(1)
    discount = torch.full_like(exponent, fill_value=discount, dtype=dtype, device=device)
    decay_matrix = torch.triu(torch.pow(input=discount, exponent=exponent), diagonal=0)
    """
    [1.0000, 0.9900, 0.9801, 0.9703, 0.9606],
    [0.0000, 1.0000, 0.9900, 0.9801, 0.9703],
    [0.0000, 0.0000, 1.0000, 0.9900, 0.9801],
    [0.0000, 0.0000, 0.0000, 1.0000, 0.9900],
    [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
    """
    
    # 计算折扣累积和
    return torch.matmul(decay_matrix, rewards.unsqueeze(-1)).squeeze(-1)

def batch_gae(rew, val, gamma=0.99, lam=1.0):
    """
    批处理版本的 GAE (Generalized Advantage Estimation) 计算
    
    参数:
        rew (Tensor): 奖励张量，形状为 [batch_size, seq_len]
        val (Tensor): 状态值函数张量，形状为 [batch_size, seq_len]
        gamma (float): 折扣因子
        lam (float): GAE 的 lambda 参数
        
    返回:
        adv (Tensor): 优势函数，形状同输入
        ret (Tensor): 目标回报值，形状同输入
    """
    # Schemule 1
    # 计算 delta: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    # # 注意处理序列末尾的特殊情况
    delta = torch.zeros_like(rew)
    # delta = rew.clone()
    delta[:, :-1] = rew[:, :-1] + gamma * val[:, 1:] - val[:, :-1]
    # 对每个序列进行折扣累积和计算
    adv = discount_cumsum_batched(delta, discount=gamma * lam)
    # 最后一个时间步的 delta 特殊处理
    adv[:, -1] = rew[:, -1] - val[:, -1]
    
    
    # 计算目标回报: R_t = A_t + V(s_t)
    ret = adv + val
    
    return adv, ret

def compute_correctness(gap, th=0.5, th_min=0.07):
    """
    批处理计算正确性分数
    
    参数:
        gap (Tensor): 预测值与目标值的差距，形状为 [batch_size, chain_len]
        th (float): 阈值，默认为 0.5
        
    返回:
        is_correct (Tensor): 正确性分数，形状与 gap 相同
    """
    # 创建条件掩码
    mask_under_th = gap < th
    mask_small_gap = gap <= th_min
    
    # 初始化结果张量
    is_correct = torch.full_like(gap, -0.1, device=gap.device)  # 默认值 -0.1
    
    # 当 gap < th 时
    is_correct[mask_under_th] = 0.1
    
    # 当 gap <= 0.07 时，覆盖之前的赋值
    is_correct[mask_small_gap] = 1.0
    
    return is_correct


class ValueModel(nn.Module):
    def __init__(self, policy_dim=512):
        super().__init__()
        policy_emb = policy_dim
        self.head = nn.Linear(policy_emb, 1)
        nn.init.normal_(self.head.weight, mean=0., std=1/policy_emb)
        nn.init.constant_(self.head.bias, 0.)
        
    def forward(self, x):
        x = self.head(x)
        return x


class Detector(nn.Module):
    def __init__(self, model=None, model_scale="ViT-B/16"):
        super(Detector,self).__init__()
        
        self.image_encoder = Image_Encoder_V2(model)
        
        if model_scale == "ViT-B/16":
            self.vm_head = nn.Linear(512, 2)
            self.value_head = nn.Linear(512, 1)
        else:
            self.vm_head = nn.Linear(768, 2)
            self.value_head = nn.Linear(768, 1)

    
    def test_forward(self, x):
        cls_features, v_feat = self.image_encoder(x)
        probs = self.vm_head(cls_features)
        if self.training:
            return probs, cls_features

        return probs
    
    def extract_features(self, x):
        cls_features, vision_tokens = self.image_encoder(x)
        return cls_features
    
    def feat_forward(self, x):
        cls_features, _ = self.image_encoder(x)
        probs = self.vm_head(cls_features)
        return probs, cls_features
    
    
    def forward(self, x, cod_times=5):
        cls_features, v = self.image_encoder(x[:, :, :, :, 0])
        value = self.value_head(cls_features)
        # value = self.value_head(v)
        values_list = value.unsqueeze(1)
        probs = self.vm_head(cls_features)
        probs_list = probs.unsqueeze(1)
        
        for i in range(1, cod_times):
            cls_features, v = self.image_encoder(x[:, :, :, :, i])
            value = self.value_head(cls_features)
            # value = self.value_head(v)
            values_list = torch.cat([values_list, value.unsqueeze(1)], dim=1)
            probs = self.vm_head(cls_features)
            probs_list = torch.cat([probs_list, probs.unsqueeze(1)], dim=1)
            

        return probs_list, values_list[..., 0]
    

    

if __name__ == '__main__':
    ## CLIP part ##
    from CLIP import clip
    clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")   #ViT-B/16 #ViT-B/32
    x = torch.randn([32, 3, 224, 224]).cuda()
    y = torch.ones([4, 5]).cuda()
    x = torch.randn([2, 3, 224, 224, 5]).cuda()
    model = Detector(clip_model).cuda()
    model.value_head = ValueModel().cuda()
    model.train()
    
    # with torch.inference_mode():
    _ = model(x)
    model.eval()
    model.test_forward(x[:, :, :, :, 0])
    
    
    print("ok")
    
    

