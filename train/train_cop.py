import gc
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from sklearn import metrics
from torch.utils import data
import os
from test_loader import Get_DataLoader
from FFPP_Dataset import FFPP_ReFT_Dataset


seed = 1024
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
print('Random seed :', seed)
# torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# torch.autograd.set_detect_anomaly(True)

# import cv2
# def Tensor2cv(img_tensor):
#     img_tensor *= torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(-1).unsqueeze(-1).cuda()
#     img_tensor += torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(-1).unsqueeze(-1).cuda()
#     img_tensor = img_tensor.detach().permute(1, 2, 0).cpu()
#     img_numpy = img_tensor.numpy() * 255
#     img_numpy = np.uint8(img_numpy)
#     img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
#     return img_numpy

# img1 = Tensor2cv(img[0, ..., 0])
# cv2.imwrite("test.png", img1)
# exit(0)

from collections import OrderedDict
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def compute_AUC(y, pred):
    auc = metrics.roc_auc_score(y, pred)
    return auc



device = "cuda" if torch.cuda.is_available() else "cpu"
from model_zoo import Detector, logprobs_from_logits, ValueModel, batch_gae, compute_correctness
from CLIP import clip
# model_scale = "ViT-L/14"  # ViT-B/16  ViT-L/14
model_scale = "ViT-B/16" 
clip_model, preprocess = clip.load(model_scale, device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")
model = Detector(clip_model, model_scale).cuda()

if model_scale == "ViT-B/16":
    model_scale = "ViT_B_16"
    patch_num = 14
    batch_size = 8
    gradient_accumulation_steps = 1  # 累积n个batch的梯度后再更新参数
    path = "/home/liu/fcb1/reft/trained_weight/A_SL_clip_ViT_B_16_Epoch1_cdf0.7976.pth"
    # path = "/home/liu/fcb1/reft/trained_weight/ABL_warmup3_SL_clip_ViT_B_16_Epoch3_cdf0.7866.pth"
    model.value_head = ValueModel(policy_dim=512).to(device)
    best_auc = 0.7
else:
    model_scale = "ViT_L_14"
    patch_num = 16
    batch_size = 2
    gradient_accumulation_steps = 4 # bset
    path = "/home/liu/fcb1/reft/trained_weight/A_SL_clip_ViT_L_14_Epoch1_cdf0.8884.pth"
    model.value_head = ValueModel(policy_dim=768).to(device)
    best_auc = 0.91

model.load_state_dict(torch.load(path)['model'], strict=False)
print(f"loaded SL pretrained weights from {path}.")


from copy import deepcopy
ref_model = deepcopy(model).to(device)  # Create an EMA of the model for use after training
update_ema(ref_model, model, decay=0)  # Ensure EMA is initialized with synced weights
requires_grad(ref_model, False)
ref_model.eval()

image_size = 224
train_dataset = FFPP_ReFT_Dataset(phase='train',image_size=image_size, compress='c23', tamper='DF', n_frames=8, patch_num=patch_num)
train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn,
                                            drop_last=True
                                            )


TestSet = Get_DataLoader(dataset_name="VideoCDF", image_size=224, 
                          normalize='clip'
                        #  normalize="imagenet"
                        #  normalize=False
                         )

train_len = len(train_set)
print('length of TrainSet is ', train_len)

# -----------------Build optimizerr-----------------

model_params = [
{'params': model.parameters(), 'lr': 3e-7, 'weight_decay': 1e-7, 'max_grad_norm':1.}, # best
]



optims = 'adan'
# optims = "adamw"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), eps=1e-6)
elif optims == 'adamw':
    optimizer = optim.AdamW(model_params, betas=(0.9, 0.999), eps=1e-6)

print('Current Optimizer is', optims)

Epoch = 100

device = torch.device('cuda')
gc.collect()
torch.cuda.empty_cache()
print("Let's start training!")

# CLIP-ViT  
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1, 1).cuda()
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1, 1).cuda()


e_cnt = 1
train_start_time = time.time()

ppo_epoches = 2


@torch.no_grad()
def rollout(model, img, chain_len, gt_lab):
    model.eval()
    # with torch.no_grad():
    probs_list, val = model(img, chain_len)   # old model
    old_logprob = logprobs_from_logits(probs_list, labels=gt_lab)
    
    # Get the ref model logprob
    ref_probs_list, _ = ref_model(img, chain_len)
    ref_logprob = logprobs_from_logits(ref_probs_list, labels=gt_lab)

    # Evaluate score
    # threshold for binary classification
    # 计算每个时间步的预测值和目标值
    final_answers = probs_list.softmax(dim=-1)[:, :, 1]  # 获取每个时间步的最终预测值

    # 计算所有时间步的差距 (gap)
    gap = torch.abs(final_answers - gt_lab)
    
    # 计算正确性分数 (correctness)
    # best
    th = 0.5 
    th_min = 0.07
    
    # th = 0.5 
    # th_min = 0.1
    
    correctness = compute_correctness(gap, th, th_min)

    rew = correctness.clone()

    # # best
    gamma = 0.99
    lam = 1.
    
    adv, ret = batch_gae(rew, val, gamma, lam)

    model.train()
    return rew, correctness, ret, val, old_logprob, ref_logprob, adv, gt_lab, img


effective_batch_size = batch_size * gradient_accumulation_steps  # 实际等效batch_size
print(f"Using gradient accumulation: {gradient_accumulation_steps} steps, effective batch size = {effective_batch_size}")


for e in range(0, Epoch):
    start = time.time()
    model.train()
    
    for step_id, data in enumerate(train_set):
        img = data['img'].to(device, non_blocking=True).float()
        label = data['label'].to(device, non_blocking=True).float() 
        # prob_list = data['prob_list'].to(device, non_blocking=True).float() 
        
        img = (img - clip_mean) / clip_std
        chain_len = img.shape[-1]
        
        rew, correctness, ret, val, old_logprob, ref_logprob, adv, label, img = rollout(model, img.detach(), chain_len, label.detach())
        
        for j in range(ppo_epoches):
            
            cur_img = img
            cur_label = label
            cur_ret = ret
            cur_val = val
            cur_old_logprob = old_logprob
            cur_adv = adv
            cur_ref_logprob = ref_logprob
        
            cur_probs, cur_values = model(cur_img, chain_len)
            logprob = logprobs_from_logits(cur_probs, cur_label)
            prob_loss = 0.5 * F.cross_entropy(cur_probs.view(-1, 2), cur_label.view(-1).long().detach()) # best


            loss = 0.
            # policy gradient loss
            ratio = torch.exp(logprob - cur_old_logprob)
            kl_ref = torch.exp(cur_ref_logprob-logprob) - torch.log(torch.exp(cur_ref_logprob-logprob)) - 1.
            beta = 0.4 # best


            clip_lower = 0.3
            clip_higher = 0.3
            pg_losses = -cur_adv * ratio
            pg_losses2 = -cur_adv * torch.clamp(ratio, 1.0 - clip_lower, 1.0 + clip_higher)
            pg_loss = ((torch.max(pg_losses, pg_losses2) + beta * kl_ref).sum(dim=-1) / chain_len).mean()

            # value loss
            clip_lower = 0.3
            clip_higher = 0.3
            vpredclipped = torch.clamp(cur_values, cur_val - clip_lower, cur_val + clip_higher)
            vf_losses1 = (cur_values - cur_ret) ** 2
            vf_losses2 = (vpredclipped - cur_ret) ** 2
            vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2)).sum(dim=-1) / chain_len).mean()

            # total loss
            # loss += 1. * pg_loss + 1. * vf_loss + 0.2 * prob_loss 
            loss += 1. * pg_loss + 1. * vf_loss ## best for ViT-B/16 (CE loss)
            # loss += 1. * pg_loss + 5. * vf_loss + 0.5 * prob_loss # best for ViT-L/14
            # loss += 1. * pg_loss + 5. * vf_loss
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            # 达到累积步数时更新参数
            if (step_id + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度

            
        if not (step_id+1) % (30 * gradient_accumulation_steps):
            print(f"epoch: {e} / {Epoch},step {step_id+1} / {len(train_set)}, loss: {loss.item()*gradient_accumulation_steps:.6f}")
            print(f"pg loss: {pg_loss.item():.6f}; vf loss: {vf_loss.item():.6f}; ", end='',flush=True)
            print(f"kl_ref: {kl_ref.mean().item():.6f}; score_rew: {rew.mean().item():.6f}; ", end='',flush=True)
            print(f"prob loss {prob_loss.item():.6f}", end='',flush=True)
            print('\n', end='',flush=True)



        if (step_id+1) % (90*gradient_accumulation_steps) == 0:
            
            model.eval()
            outputs = None
            testargets = None
            print('testing......')
            with torch.inference_mode():
                for step_id, datas in enumerate(TestSet):
                    img = datas[0].cuda()
                    targets = datas[1].cuda()
                    output = model.test_forward(img)
                    cls_final = torch.softmax(output, dim=-1)[:, 1]

                    
                    n_frames = img.shape[0]
                    targets = targets.expand(n_frames, -1)
                    outputs = cls_final if outputs is None else torch.cat(
                        (outputs, cls_final), dim=0)
                    testargets = targets if testargets is None else torch.cat(
                        (testargets, targets), dim=0)
            cur_auc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach())
            print(f'Frame-level CDF test AUC:{cur_auc:.4f}')
            
            if best_auc < cur_auc:
                # best_auc = cur_auc
                checkpoint = {
                        "epoch": e_cnt,
                        "model": model.state_dict(),
                    }
                torch.save(checkpoint, f'/home/liu/fcb1/reft/trained_weight/ABL_rew_0.5_0.1_RL_{model_scale}_Epoch{e_cnt}_cdf{cur_auc:.4f}.pth')

    
            end = time.time()
            print(f"epoch: {e_cnt} end ; cost time: {(end - start)/60.:.4f} min")
            start = time.time()
            e_cnt += 1
            model.train()

          
print('train ended !')