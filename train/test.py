import gc
import time
import torch
from sklearn import metrics
import random
import numpy as np
import cv2
import random
from metrics_util import *
from test_loader import Get_DataLoader


seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# CUDA_LAUNCH_BLOCKING=1

def Tensor2cv(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0).cpu()
    img_numpy = img_tensor.numpy() * 255
    img_numpy = np.uint8(img_numpy)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_numpy


img_size = 224
device = "cuda" if torch.cuda.is_available() else "cpu"
from CLIP import clip
# model_scale = "ViT-L/14" # "ViT-L/14" "ViT-B/16" ViT-L/14@336px
model_scale = "ViT-B/16"
clip_model, preprocess = clip.load(model_scale, 
    device=device, download_root="/home/liu/fcb1/clip_model")#ViT-B/16


from model_zoo import Detector
model = Detector(clip_model, model_scale).to('cuda')

if model_scale == "ViT-B/16":
    wp = "/home/liu/fcb1/reft/trained_weight/ABL_rew_0.5_0.1_RL_ViT_B_16_Epoch31_cdf0.8617.pth" 
    # wp = "/home/liu/fcb1/reft/trained_weight/ABL_warmup3_RL_ViT_B_16_Epoch6_cdf0.8300.pth"
    # wp = "/home/liu/fcb1/reft/trained_weight/A_RL_ViT_B_16_Epoch31_cdf0.8601_wCE_0.1.pth"
    model.load_state_dict(torch.load(wp)['model'], strict=False) 
elif model_scale == "ViT-L/14":
    wp = "/home/liu/fcb1/reft/trained_weight/ABL_lam_1_0_RL_ViT_L_14_Epoch44_cdf0.9226.pth"
    model.load_state_dict(torch.load(wp)['model'], strict=False)
print(f"Loaded from {wp}...")

'''
通过idx判断使用VideoCDF、VideoDFDC还是VideoDFV1、VideoDFD、VideoDFDCP
若idx为False,则测试VideoDFV1、VideoDFD、VideoDFDCP
若idx为True,则测试VideoCDF、VideoDFDC
'''
test_name = "VideoCDF" # 518
# test_name = "VideoDFDC" # 4986
# test_name = 'VideoDFV1' # 2412
# test_name = 'VideoDFD' # 4068
test_name = 'VideoDFDCP' # 777
# test_name = 'WildDeepFake' # 811
if test_name in ["VideoCDF","VideoDFDC","VideoDFD", "VideoDFV1"]:
    use_idx = True
if test_name in ["VideoDFDCP", 'WildDeepFake']:
    use_idx = False

 
TestSet = Get_DataLoader(dataset_name=test_name,
                          image_size=img_size,
                        #   normalize='imagenet'
                          normalize = 'clip',
                        #   normalize=False,
                          )

# frame_level = False
frame_level = True

if frame_level:
    print('testing in frame level now!')
else:
    print('testing in video level now!')

model.eval()
outputs = None
testargets = None
print('start testing...')
print(len(TestSet))
cnt = 0
feats = None
time_list = []
with torch.no_grad():
    start = time.time()
    for step_id, datas in enumerate(TestSet):
        if datas[0] is None:
            continue
        img = datas[0].cuda()
        targets = datas[1].cuda()
        if use_idx:
            idx_path = datas[2]
            idx_list = np.load(idx_path).tolist()  
        # output = model.test_forward(img)
        output, feat = model.feat_forward(img)
        # output, feat = model.infer_fwd(img)
        # feat = feat / feat.norm(dim=1, keepdim=True)
        # cls_final = torch.sigmoid(output)
        cls_final = torch.softmax(output, dim=-1)[:, 1:].cpu()
        # cls_final = output.cpu()
        
        
        if use_idx:
            pred_list=[]
            feat_list=[]
            idx_img=-1
            for i in range(len(cls_final)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    feat_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(cls_final[i].item())
                feat_list[-1].append(feat[i, :].cpu().numpy())

            pred_res=np.zeros(len(pred_list))
            pred_res_feat=np.zeros([len(feat_list), feat.shape[-1]])
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
                
                idx = int(np.argmax(pred_list[i]))
                pred_res_feat[i] = feat_list[i][idx]
                # print(feat_batch)
            if frame_level:
                pred=pred_res
                
                n_frames = len(pred)
                targets = targets.expand(n_frames, -1)
                feat_batch = torch.tensor(pred_res_feat, dtype=torch.float32)
                feats = feat_batch if feats is None else torch.cat((feats, feat_batch), dim=0)

            else:
                pred=pred_res.mean()
            cls_final = torch.tensor(pred, dtype=torch.float32).unsqueeze(-1).cpu()
            
        else:
            if frame_level:
                n_frames = img.shape[0]
                targets = targets.expand(n_frames, -1)
                feats = feat if feats is None else torch.cat((feats, feat), dim=0)
            else:
                cls_final = cls_final.mean().unsqueeze(0)
                feat = feat.mean(dim=0, keepdim=True)
                feats = feat if feats is None else torch.cat((feats, feat), dim=0)
        outputs = cls_final if outputs is None else torch.cat((outputs, cls_final), dim=0)
        testargets = targets if testargets is None else torch.cat((testargets, targets), dim=0)
        
    
        if not (step_id+1) % 100:
            now_percent = int(step_id / len(TestSet) * 100)
            print(f"Test: complete {now_percent} %")

'''
在图像层面保存t_SNE
'''
# from tsne_tool import save_tSNE
# save_tSNE(feats.detach().cpu().numpy(), testargets.detach().cpu(), save_name=test_name)

cdfauc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1)
ap = metrics.average_precision_score(testargets.cpu().detach(), outputs.cpu().detach())
eer = compute_EER(testargets.cpu().detach(), outputs.cpu().detach())
print(f'{test_name} test AUC:{cdfauc:.4f}; AP: {ap:.4f}; EER: {eer:.4f}')
'''
t对于acc、f1、recall、precision都有影响
'''

t_list = [0.1, 0.2, 0.3, 0.4, 0.5]
y_true = testargets.cpu().numpy()
y_pred = outputs.cpu().numpy()
for t in t_list:
    print('threshold :', t)
    acc = compute_ACC(y_true.copy(), y_pred.copy(), n_class=1, t=t)
    f1 = compute_F1(y_true.copy(), y_pred.copy(), n_class=1, t=t)
    recall = compute_recall(y_true.copy(), y_pred.copy(), n_class=1, t=t)
    pre = compute_precision(y_true.copy(), y_pred.copy(), n_class=1, t=t)
    print(f'acc : {acc:.4f} ; f1 : {f1:.4f} ; recall : {recall:.4f} ; precision : {pre:.4f}')



end = time.time()
spent = (end - start) / 60
print(f'spent time: {spent:.4f} min')
print('ending')
gc.collect()
torch.cuda.empty_cache()
exit(0)


