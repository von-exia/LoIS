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
from FFPP_Dataset import FFPP_Dataset_V2


seed = 1024
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print('Random seed :', seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.autograd.set_detect_anomaly(True)

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

def compute_AUC(y, pred, n_class=1):
    # compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    # compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc

                

device = "cuda" if torch.cuda.is_available() else "cpu"
from model_zoo import Detector

from CLIP import clip
model_scale = "ViT-L/14"  # ViT-B/16 ViT-L/14
model_scale = "ViT-B/16" 
clip_model, preprocess = clip.load(model_scale,  
device=torch.device("cpu"), download_root="./clip_model")

model = Detector(clip_model, model_scale).cuda().float()

if model_scale == "ViT-B/16":
    model_scale = "ViT_B_16"
    patch_num = 14
    batch_size = 32
    gradient_accumulation_steps = 1  
else:
    model_scale = "ViT_L_14"
    patch_num = 16
    batch_size = 8
    gradient_accumulation_steps = 4




image_size = 224
train_dataset = FFPP_Dataset_V2(phase='train',image_size=image_size, compress='c23', tamper='all', patch_num=patch_num)
train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size//2,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=4,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn,
                                            drop_last=True
                                            )


TestSet = Get_DataLoader(dataset_name="VideoCDF", image_size=image_size, 
                         normalize='clip'
                        #  normalize="imagenet"
                        #  normalize=False
                         )

train_len = len(train_set)
print('length of TrainSet is ', train_len)

# -----------------Build optimizerr-----------------
best_auc = 0.7

ie_lr = 5e-7
model_params = [
{'params': model.image_encoder.parameters(), 'lr': 5e-7, 'weight_decay': 1e-2, 'max_grad_norm':1.}, 
{'params': model.vm_head.parameters(), 'lr': 1e-3, 'weight_decay': 5e-2, 'max_grad_norm':1.}, 
]


optims = 'adan'
# optims = "adamw"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), eps=1e-6)
elif optims == 'adamw':
    optimizer = optim.AdamW(model_params, betas=(0.9, 0.999), eps=1e-8)
print('Current Optimizer is', optims)


device = torch.device('cuda')
gc.collect()
torch.cuda.empty_cache()
print("Let's start training!")

# CLIP
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).cuda()
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).cuda()


effective_batch_size = batch_size * gradient_accumulation_steps 
print(f"Using gradient accumulation: {gradient_accumulation_steps} steps, effective batch size = {effective_batch_size}")

max_val_auc = 0.
train_start_time = time.time()
Epoch = 30
cdf_aucs = []
auc_cnt = 0
for e in range(1, Epoch+1):
    start = time.time()
    model.train()
    
    for step_id, data in enumerate(train_set):
        img = data['img'].to(device, non_blocking=True).float()
        label = data['label'].to(device, non_blocking=True).long()
        img = (img - clip_mean) / clip_std
        
        probs, _ = model.test_forward(img)  
        loss = F.cross_entropy(probs, label)
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if (step_id + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad() 

        
        if not (step_id+1) % (30 * gradient_accumulation_steps):
            print(f"epoch: {e} / {Epoch},step {step_id+1} / {len(train_set)}, loss: {loss.item() * gradient_accumulation_steps:.4f}")

        if not (step_id+1) % (90 * gradient_accumulation_steps):
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
                        "epoch": e,
                        "model": model.state_dict(),
                    }
                torch.save(checkpoint, f'./trained_weight/SL_clip_{model_scale}_Epoch{e}_cdf{cur_auc:.4f}.pth')
            model.train()
    

    end = time.time()
    print(f"epoch: {e} end ; cost time: {(end - start)/60.:.4f} min")
    start = time.time()
    model.train()

print('train ended !')
