import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
import torchvision.transforms as tr
from glob import glob
import os
import numpy as np
import random
import cv2
import json
import pandas as pd
from PIL import Image
import time
from torchvision.transforms import InterpolationMode
import sys
import warnings
import albumentations as alb
warnings.filterwarnings('ignore')
import json


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])
    
    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4#0#np.random.rand()*(w/8)
        w1_margin=w/4
        h0_margin=h/4#0#np.random.rand()*(h/5)
        h1_margin=h/4
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/32#w/8#0#np.random.rand()*(w/8)
        w1_margin=w/32#w/8
        h0_margin=h/4.5#h/2#0#np.random.rand()*(h/5)
        h1_margin=h/16# h/5

    if margin:
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        # print('dynamic cropping')
        # SBI default for EFB4
        # w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        # w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        # h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        # h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        
        w0_margin*=(np.random.rand()*.1+3.95)#np.random.rand()
        w1_margin*=(np.random.rand()*.1+3.95)#np.random.rand()
        h0_margin*=(np.random.rand()*.1+1.95)#np.random.rand()
        h1_margin*=(np.random.rand()*.1+1.95)#np.random.rand()
     
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
            
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)
    
    img_cropped=img[y0_new:y1_new,x0_new:x1_new]
    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

def init_fff(phase, dataset_paths, DF, NT, FS, FF, n_frames=8):
    landmark_path = '/home/liu/fcb1/dataset/FFPP_16/landmarks/'
    image_list = []
    DFs_list = []
    NTs_list = []
    FSs_list = []
    FFs_list = []
    landmark_list=[]
    folder_list = sorted(glob(os.path.join(dataset_paths,'*')))
    landmark_folder_list = sorted(glob(landmark_path+'*'))
    DF_list = sorted(glob(os.path.join(DF,'*')))
    DF_list = sorted(DF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    NT_list = sorted(glob(os.path.join(NT,'*')))
    NT_list = sorted(NT_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FS_list = sorted(glob(os.path.join(FS,'*')))
    FS_list = sorted(FS_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FF_list = sorted(glob(os.path.join(FF,'*')))
    FF_list = sorted(FF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    filelist = []
    list_dict = json.load(open(f'/home/liu/fcb1/dataset/FFplus/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    landmark_folder_list = [i for i in landmark_folder_list if os.path.basename(i)[:3] in filelist]
    DF_list = [i for i in DF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    NT_list = [i for i in NT_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FS_list = [i for i in FS_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FF_list = [i for i in FF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    for i in range(len(folder_list)):
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        DF_temp = sorted(glob(DF_list[i]+'/*.png'))
        NT_temp = sorted(glob(NT_list[i]+'/*.png'))
        FS_temp = sorted(glob(FS_list[i]+'/*.png'))
        FF_temp = sorted(glob(FF_list[i]+'/*.png'))
        landmarks_temp=sorted(glob(landmark_folder_list[i]+'/*.npy'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
            DF_temp=[DF_temp[round(i)] for i in np.linspace(0,len(DF_temp)-1,n_frames)]
            NT_temp=[NT_temp[round(i)] for i in np.linspace(0,len(NT_temp)-1,n_frames)]
            FS_temp=[FS_temp[round(i)] for i in np.linspace(0,len(FS_temp)-1,n_frames)]
            FF_temp=[FF_temp[round(i)] for i in np.linspace(0,len(FF_temp)-1,n_frames)]
            landmarks_temp=[landmarks_temp[round(i)] for i in np.linspace(0,len(landmarks_temp)-1,n_frames)]     
        image_list+=images_temp
        DFs_list+=DF_temp
        NTs_list+=NT_temp
        FSs_list+=FS_temp
        FFs_list+=FF_temp
        landmark_list+=landmarks_temp
    return image_list,DFs_list,NTs_list,FSs_list,FFs_list,landmark_list

# def landmark_fff(phase, n_frames=8):
#     print(phase)
#     landmark_path = '/home/liu/fcb1/dataset/FFPP_16/landmarks/'
#     landmark_list=[]
#     folder_list = sorted(glob(landmark_path+'*'))
#     filelist = []
#     list_dict = json.load(open(f'/home/liu/fcb1/dataset/FFplus/{phase}.json','r'))
#     for i in list_dict:
#         filelist+=i
#     folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
#     for i in range(len(folder_list)):
#         landmarks_temp=sorted(glob(folder_list[i]+'/*.npy'))
#         landmark_list+=landmarks_temp
#     return landmark_list

class FFPP_ReFT_Dataset(Dataset):
    def __init__(self,compress='raw',image_size=224,phase = "train", tamper="all", n_frames=8, patch_num=14):# raw、c23、c40
        super().__init__()
        self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/{compress}/images/'
        self.Deepfakes_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/home/liu/sdb/FFPP/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/home/liu/sdb/FFPP/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Face2Face/{compress}/images/'
        self.patch_num = patch_num
 
        print('start')
        st = time.time()
        if phase == 'val' or phase == 'test':
            n_frames = 4
        else:
            n_frames = n_frames
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=n_frames)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
            
            
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))
        

        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        self.img_size = image_size
        
        self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.augmentation = self.get_transforms()
        self.real_augmentation = self.get_real_transforms()
        self.phase = phase
        self.tamper = tamper
    
    def __len__(self):
        return len(self.fake_frame_list)
    

    def get_transforms(self):
        return alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RGBShift((-20,20),(-20,20),(-20,20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2),
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),
            alb.RandomGridShuffle((self.img_size//self.patch_num, self.img_size//self.patch_num), p=0.15), 
        ], 
        additional_targets={f'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'},
        p=1.)
        
    def get_real_transforms(self):
        return alb.Compose([
            alb.HorizontalFlip(p=0.5),

            alb.RGBShift((-10,10),(-10,10),(-10,10), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.1,0.1), sat_shift_limit=(-0.1,0.1), val_shift_limit=(-0.1,0.1), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=0.3),
            
            alb.ImageCompression(quality_lower=70,quality_upper=100,p=0.2),
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),
            # alb.RandomGridShuffle((self.img_size//16, self.img_size//16), p=0.15),
            alb.RandomGridShuffle((self.img_size//self.patch_num , self.img_size//self.patch_num ), p=0.15), 
        ], 
        additional_targets={f'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'},
        p=1.)
    
    def __getitem__(self, idx):
        # print(idx)
        
        flag = True
        while flag:
            try:
                img_path = self.fake_frame_list[idx]
                real_id = img_path.split('/')[-2].split('_')[0]
                frame_id = img_path.split('/')[-1].split('.')[0] + '.npy'
                lamk_path = self.landmark_root + real_id + '/' + frame_id
                png_path = img_path.split('images')[-1][1:]
                forgery_type = img_path.split('/')[-5]
                # fake_img = cv2.imread(img_path)
                # fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                
                real_img_path = self.real_root + real_id + '/' + frame_id.replace(".npy", ".png")
                img = cv2.imread(real_img_path)
                real_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # if ms_blend:
                df_path = self.Deepfakes_root + png_path
                fs_path = self.FaseSwap_root + png_path 
                ff_path = self.Face2Face_root + png_path 
                nt_path = self.NeuralTextures_root + png_path
        
                
                df_img = cv2.imread(df_path)
                df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
                fs_img = cv2.imread(fs_path)
                fs_img = cv2.cvtColor(fs_img, cv2.COLOR_BGR2RGB)
                ff_img = cv2.imread(ff_path)
                ff_img = cv2.cvtColor(ff_img, cv2.COLOR_BGR2RGB)
                nt_img = cv2.imread(nt_path)
                nt_img = cv2.cvtColor(nt_img, cv2.COLOR_BGR2RGB)
                
                # prob_r = [1., 0., 0., 0., 0.]
                # prob_1 = [0., 1., 0., 0., 0.]
                # prob_2 = [0., 0., 1., 0., 0.]
                # prob_3 = [0., 0., 0., 1., 0.]
                # prob_4 = [0., 0., 0., 0., 1.]
                
                # p = 0.1 # best
                # if np.random.randn() < p:
                #     df_img, _, prob_1 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < p:
                #     fs_img, _, prob_2 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < p:
                #     ff_img, _, prob_3 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < p:
                #     nt_img, _, prob_4 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                    
                # if np.random.randn() < 0.1:
                #     df_img, _, prob_1 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < 0.2:
                #     fs_img, _, prob_2 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < 0.3:
                #     ff_img, _, prob_3 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                # if np.random.randn() < 0.4:
                #     nt_img, _, prob_4 = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                    
                    # fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending.png", np.uint8(fake_img))
                    # FI_map = cv2.cvtColor(np.abs(FI_map), cv2.COLOR_RGB2BGR) * 10
                    # FI_map[FI_map>255]=255
                    # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending_FT_map.png", np.uint8(FI_map))
                    # exit(0)
                # else:
                #     if forgery_type == "Deepfakes":
                #         soft_label_list = [1., 0., 0., 0.]
                #     elif forgery_type == "FaceSwap":
                #         soft_label_list = [0., 1., 0., 0.]
                #     elif forgery_type == "Face2Face":
                #         soft_label_list = [0., 0., 1., 0.]
                #     elif forgery_type == "NeuralTextures":
                #         soft_label_list = [0., 0., 0., 1.]
                
                
                landmark=np.load(lamk_path)[0]
                landmark = self.reorder_landmark(landmark)
                flag = False
            except Exception as e:
                # print(e)
                idx = torch.randint(low=0,high=len(self.fake_frame_list),size=(1,)).item()
        

        if self.phase == 'train':
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
            # real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
        else:
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)

        # real_img = self.trans(real_img)  # divide 255
    
        # fake_img = fake_img[y0_new:y1_new,x0_new:x1_new]
        df_img = df_img[y0_new:y1_new,x0_new:x1_new]
        fs_img = fs_img[y0_new:y1_new,x0_new:x1_new]
        ff_img = ff_img[y0_new:y1_new,x0_new:x1_new]
        nt_img = nt_img[y0_new:y1_new,x0_new:x1_new]
        
        # if np.random.rand() < .5:
        #     c_flag = True
        #     while c_flag:
        #         try:
        #             path = self.celeba_img[self.c_id]
        #             lmk_path = self.celeba_lamk[self.c_id]
        #             lmk = np.load(lmk_path)
        #             real_img = cv2.imread(path)
        #             real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        #             real_img, _, _, _, _, _, _, _ = crop_face(real_img,lmk,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
        #             self.c_id = self.c_id + 1 if self.c_id < len(self.celeba_img) else 0
        #             c_flag = False
        #         except Exception as e:
        #             # print(e)
        #             self.c_id = self.c_id + 1 if self.c_id < len(self.celeba_img) else 0
        
        inter_flag = cv2.INTER_CUBIC
        real_img = cv2.resize(real_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        # fake_img = cv2.resize(fake_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        df_img = cv2.resize(df_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        fs_img = cv2.resize(fs_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        ff_img = cv2.resize(ff_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        nt_img = cv2.resize(nt_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        
        
        # cv2.imwrite("/home/liu/fcb1/reft/visual/real.png", \
        #             cv2.cvtColor(real_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("/home/liu/fcb1/reft/visual/df_fake.png", \
        #     cv2.cvtColor(df_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("/home/liu/fcb1/reft/visual/fs_fake.png", \
        #     cv2.cvtColor(fs_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("/home/liu/fcb1/reft/visual/ff_fake.png", \
        #     cv2.cvtColor(ff_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("/home/liu/fcb1/reft/visual/nt_fake.png", \
        #     cv2.cvtColor(nt_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # exit(0)
        
        
        # real_img1 = self.real_augmentation(image=real_img.copy().astype('uint8'))['image'].transpose((2, 0, 1)) / 255.
        # real_img2 = self.real_augmentation(image=real_img.copy().astype('uint8'))['image'].transpose((2, 0, 1)) / 255.
        # real_img3 = self.real_augmentation(image=real_img.copy().astype('uint8'))['image'].transpose((2, 0, 1)) / 255.

        
        
        transformed=self.augmentation(image=real_img.astype('uint8'),\
                image1=df_img.astype('uint8'),
                image2=fs_img.astype('uint8'),
                image3=ff_img.astype('uint8'),
                image4=nt_img.astype('uint8'),
                )
        real_img=transformed['image'].transpose((2, 0, 1)) / 255.
        df_img=transformed['image1'].transpose((2, 0, 1)) / 255. 
        fs_img=transformed['image2'].transpose((2, 0, 1)) / 255. 
        ff_img=transformed['image3'].transpose((2, 0, 1)) / 255. 
        nt_img=transformed['image4'].transpose((2, 0, 1)) / 255. 
        
        img_list = [real_img, df_img, fs_img, ff_img, nt_img] # best, but why
        lab_list = [0., 1., 1., 1., 1.]   
        
        # img_list = [df_img, fs_img, ff_img, nt_img, real_img] 
        # lab_list = [1., 1., 1., 1., 0.]   

        # img_list = [real_img, df_img, real_img1, fs_img, real_img2, ff_img, real_img3, nt_img]
        # lab_list = [0., 1., 0., 1., 0., 1., 0., 1.]   
        # act_list = [0, 1, 0, 2, 0, 3, 0, 4]
        # prob_list = [prob_r, prob_1, prob_r, prob_2, prob_r, prob_3, prob_r, prob_4]
        
        # fake_list = [df_img, fs_img, ff_img, nt_img]
        # indices = list(range(len(fake_list)))
        # random.shuffle(indices)
        # fake_list = [fake_list[i] for i in indices]
        # img_list += fake_list
        # # p = np.random.rand()
        # bnd = 1 / 6
        # if p < bnd:
        #     fake_list = [df_img, fs_img]
        # elif bnd <= p < bnd*2:
        #     fake_list = [df_img, ff_img]
        # elif bnd*2 <= p < bnd*3:
        #     fake_list = [df_img, nt_img]
        # elif bnd*3 <= p < bnd*4:
        #     fake_list = [fs_img, ff_img]
        # elif bnd*4 <= p < bnd*5:
        #     fake_list = [fs_img, nt_img]
        # elif p >= bnd*5:
        #     fake_list = [ff_img, nt_img]
        
        
        # transformed=self.augmentation(image=real_img.copy().astype('uint8'),\
        #         image1=fake_list[0].astype('uint8'),
        #         )
        # real_img1=transformed['image'].transpose((2, 0, 1)) / 255.
        # fake_img1=transformed['image1'].transpose((2, 0, 1)) / 255. 
        
        # transformed=self.augmentation(image=real_img.copy().astype('uint8'),\
        #         image1=fake_list[1].astype('uint8'),
        #         )
        # real_img2=transformed['image'].transpose((2, 0, 1)) / 255.
        # fake_img2=transformed['image1'].transpose((2, 0, 1)) / 255. 

        # img_list = [real_img, fake_list[0], fake_list[1], real_img1, fake_list[2]]
        # lab_list = [0., 1., 1., 0., 1.] 
        
        # fake_lab1 = np.clip(np.random.uniform(0.99, 1.0001), 0.99, 1)
        # fake_lab2 = np.clip(np.random.uniform(0.99, 1.0001), 0.99, 1)
        # fake_lab3 = np.clip(np.random.uniform(0.99, 1.0001), 0.99, 1)
        # prob_list = [[1., 0.], 
        #             [1. - fake_lab1, fake_lab1],
        #             [1. - fake_lab2, fake_lab2],
        #             [1., 0.], 
        #             [1. - fake_lab3, fake_lab3],
        # ]
        
        # p = np.random.rand()
        # if p < 0.25:
        #     img_list = [real_img1, fake_img1, real_img2, fake_img2]
        #     lab_list = [0., 1., 0., 1.] 
        #     # act_list = [0, 1, 0, 1]
        # elif 0.25 <= p < 0.5:
        #     img_list = [fake_img1, real_img1, real_img2, fake_img2]
        #     lab_list = [1., 0., 0., 1.] 
        #     # act_list = [1, 0, 0, 1]
        # elif 0.5 <= p < 0.75:
        #     img_list = [real_img1, fake_img1, fake_img2, real_img2]
        #     lab_list = [0., 1., 1., 0.] 
        #     # act_list = [0, 1, 1, 1]
        # elif p >= 0.75:
        #     img_list = [fake_img1, real_img1, fake_img2, real_img2]
        #     lab_list = [1., 0., 1., 0.] 
        #     # act_list = [1, 0, 1, 0]
            
            
        # mask_num = random.choice([1, 2, 3])
        # padding_img = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, np.newaxis, np.newaxis]
        # padding_img = padding_img.repeat(224, axis=1)
        # padding_img = padding_img.repeat(224, axis=2)
        # padding_lab = -100
        # for i in range(1, mask_num+1):
        #     img_list[-i] = padding_img
        #     lab_list[-i] = padding_lab
        #     act_list[-i] = padding_lab
        
        # img_list = [real_img1, fake_img1, real_img2, fake_img2]
        # lab_list = [0., 1., 0., 1.] 

        # indices = list(range(len(img_list)))
        # random.shuffle(indices)
        # img_list = [img_list[i] for i in indices]
        # lab_list = [lab_list[i] for i in indices]
        # # act_list = [act_list[i] for i in indices]
        # prob_list = [prob_list[i] for i in indices]
        


        return img_list, lab_list
        
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark
    
    def collate_fn(self, batch):
        img_list, lab_list = zip(*batch)
        img_list = torch.tensor(img_list)
        img_list = img_list.permute(0, 2, 3, 4, 1)
        lab_list = torch.tensor(lab_list)
        # act_list = torch.tensor(act_list)
        # prob_list = torch.tensor(prob_list)
        
        data = {}
        data['img'] = img_list
        data['label'] = lab_list
        # data['act_list'] = act_list
        # data['prob_list'] = prob_list
        return data


    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


def randaffine(img):
    f=alb.Affine(
                translate_percent={'x':(-0.01,0.01),'y':(-0.0075,0.0075)},
                scale=[0.98,1/0.98],
                interpolation=cv2.INTER_CUBIC,
                fit_output=False,
                p=1)
            

    transformed=f(image=img)
    img=transformed['image']
    return img

from skimage.transform import PiecewiseAffineTransform, warp
def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)   # 生成坐标
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)        # 得到对应的坐标矩阵，即两个矩阵中分别保存x轴坐标与y轴坐标

    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    # if np.random.rand() < 0.5:
    #     warped = warp(image, trans, order=3).astype(np.float32)
    #     warped = warp(warped, trans.inverse, order=3).astype(np.float32)
    # else:
    warped = warp(image, trans, order=1).astype(np.float32)
    warped = warp(warped, trans.inverse, order=1).astype(np.float32)
    return warped

def get_blend_mask(mask):
    H,W=mask.shape
    # size_h=np.random.randint(192,257)
    # size_w=np.random.randint(192,257)
    size_h=np.random.randint(86,151)
    size_w=np.random.randint(86,151)
    mask=cv2.resize(mask,(size_w,size_h))
    kernel_1=random.randrange(5,26,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(5,26,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    return mask_blured.reshape((mask_blured.shape+(1,)))

def Normalize(w1, w2, w3, w4):
    _sum = np.sum([w1, w2, w3, w4])
    w1 = w1 / _sum
    w2 = w2 / _sum
    w3 = w3 / _sum
    w4 = w4 / _sum
    return w1, w2, w3, w4

def softmax(w1, w2, w3, w4):
    # 避免数值稳定性问题，减去输入中的最大值
    # a1 = np.random.randint(1, 10)
    # a2 = np.random.randint(1, 10)
    # a3 = np.random.randint(1, 10)
    # a4 = np.random.randint(1, 10)
    # x = [w1*a1, w2*a2, w3*a3, w4*a4]
    
    x = [w1*5., w2*5., w3*5., w4*5.] # best
    # exps = np.exp(x - np.max(x))
    exps = np.exp(x)
    exps = exps / np.sum(exps, axis=0)
    return exps[0], exps[1], exps[2], exps[3]


def Multiple_CutMix(fake_list):
    np.random.shuffle(fake_list)
    f1, f2, f3, f4 = fake_list
    h, w, c = f1.shape
    patch_size = 16
    patch_num = h // patch_size
    f1 = f1.astype(np.float32)
    f2 = f2.astype(np.float32)
    f3 = f3.astype(np.float32)
    f4 = f4.astype(np.float32)

    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    
    
    if np.random.rand() < 0.5:  # best
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
    else:
        w1, w2, w3, w4 = softmax(w1, w2, w3, w4)
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
        
    fake = np.zeros_like(f1)
    
    patch_num1 = patch_num**2 * w1
    low = int(np.sqrt(patch_num1))
    width1 = np.random.randint(low//4+1, patch_num-1)
    height1 = int(patch_num1 // (width1+1))
    fake[:patch_size*height1, :patch_size*width1] = f1[:patch_size*height1, :patch_size*width1]
    
    patch_num2 = patch_num**2 * w2
    width2 = patch_num - width1
    height2 = int(patch_num2 // (width2+1))
    fake[:patch_size*height2, patch_size*width1:] = f2[:patch_size*height2, patch_size*width1:]
    

    fake[patch_size*height1:, :patch_size*width1] = f3[patch_size*height1:, :patch_size*width1]
    fake[patch_size*height2:, patch_size*width1:] = f4[patch_size*height2:, patch_size*width1:]
    # fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("/home/liu/fcb1/DiM-DiffusionMamba/visualization/m_cutmix.png", np.uint8(fake))
    # exit(0)
    return fake, [w1, w2, w3, w4]
    

# import math
# import pywt
def Multiple_Soft_Blend_v2(df, fs, ff, nt, real, principal_component=None):
    real = real.astype(np.float32)
    df = df.astype(np.float32)
    fs = fs.astype(np.float32)
    ff = ff.astype(np.float32)
    nt = nt.astype(np.float32)

    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    if principal_component == "DF":
        w1 *= 2
    elif principal_component == "FS":
        w2 *= 2
    elif principal_component == "FF":
        w3 *= 2
    elif principal_component == "NT":
        w4 *= 2
    
    if np.random.rand() < 0.5:  # best
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
    else:
        w1, w2, w3, w4 = softmax(w1, w2, w3, w4)
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
    
    FI_map_df = real - df
    FI_map_fs = real - fs
    FI_map_ff = real - ff
    FI_map_nt = real - nt

    FI_map = FI_map_df * w1 + FI_map_fs * w2 + FI_map_ff * w3 + FI_map_nt * w4
    fake = real - FI_map
    fake = np.clip(fake, 0, 255)
 
    return fake, None, [w1, w2, w3, w4]

def augment_mask(mask):
    
    mask = mask.astype(np.float32)
    img_size = mask.shape[0]

    grid_size_list = [0, 2, 3, 4] # best
    grid_size = np.random.choice(grid_size_list)
    if grid_size != 0:
        lamk, lamk_deformed = random_deform((img_size, img_size), grid_size, grid_size)
        mask = piecewise_affine_transform(mask, lamk, lamk_deformed)
    else:
        mask = mask
    
    mask = randaffine(mask)
    return mask

def Multiple_Soft_Blend(df, fs, ff, nt, real):
    
    real = real.astype(np.float32)
    df = df.astype(np.float32)
    fs = fs.astype(np.float32)
    ff = ff.astype(np.float32)
    nt = nt.astype(np.float32)
    
    FI_map_df = real - df
    FI_map_fs = real - fs
    FI_map_ff = real - ff
    FI_map_nt = real - nt
    
    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    
    if np.random.rand() < 0.5:  # best
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
    else:
        w1, w2, w3, w4 = softmax(w1, w2, w3, w4)
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)

    # print(w1, w2, w3, w4)
    FI_map = FI_map_df * w1 + FI_map_fs * w2 + FI_map_ff * w3 + FI_map_nt * w4
    
    fake = real - FI_map
    fake = np.clip(fake, 0, 255)
    return fake, [FI_map_df * w1, FI_map_fs * w2, FI_map_ff * w3, FI_map_nt * w4], [w1, w2, w3, w4]


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        # ratio_list=[2]
        r=ratio_list[np.random.randint(len(ratio_list))]
        # img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_LINEAR)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds


#-------------------- For evaluation and test ------------------#
class VideoFFPP_Dataset(Dataset):
    def __init__(self,compress='raw',image_size=224,phase = "test", tamper="all", n_frames=16):# raw、c23、c40
        super().__init__()
        self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/{compress}/images/'
        self.Deepfakes_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/home/liu/sdb/FFPP/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/home/liu/sdb/FFPP/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Face2Face/{compress}/images/'
 
        print('start')
        st = time.time()
        # if phase == "test":
        #     n_frames = 2
        # else:
        #     n_frames = 4
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=n_frames)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
            # Deepfakes_list[:len(Deepfakes_list)//4] + NeuralTextures_list[len(NeuralTextures_list)//4:len(NeuralTextures_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
            # self.fake_frame_list = Deepfakes_list[:len(Deepfakes_list)//4] + FaseSwap_list[len(FaseSwap_list)//4:len(FaseSwap_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))
        
        last_name = self.real_frame_list[0].split('/')[-1]
        last_video_fold = self.real_frame_list[0].replace(last_name, "")
        self.video_list = [last_video_fold]
        self.label_list = [0]
        for i in range(1, len(self.real_frame_list)):
            cur_name =  self.real_frame_list[i].split('/')[-1]
            cur_video_fold = self.real_frame_list[i].replace(cur_name, "")
            if cur_video_fold == self.video_list[-1]:
                continue
            else:
                self.video_list.append(cur_video_fold)
                self.label_list.append(0)
        
        for i in range(0, len(self.fake_frame_list)):
            cur_name =  self.fake_frame_list[i].split('/')[-1]
            cur_video_fold = self.fake_frame_list[i].replace(cur_name, "")
            if cur_video_fold == self.video_list[-1]:
                continue
            else:
                self.video_list.append(cur_video_fold)
                self.label_list.append(1)

        # self.tamper = tamper
        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        # self.compress = compress
        self.img_size = image_size
        
        self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.phase = phase
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        
    def __len__(self):
        return len(self.video_list)    
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.label_list[index]
        
        if "Deepfakes" in video_path:
            pattern_lable = 1
        elif "FaceSwap" in video_path:
            pattern_lable = 2
        elif "Face2Face" in video_path:
            pattern_lable = 3
        elif "NeuralTextures" in video_path:
            pattern_lable = 4
        else:
            pattern_lable = 0
        
        video = None
        try:
            for _root, _, files in os.walk(video_path):
                if files:
                    for file in files:
                        # file = file.decode()
                        img_path = os.path.join(_root, file)
                        
                        real_id = img_path.split('/')[-2].split('_')[0]
                        frame_id = img_path.split('/')[-1].split('.')[0] + '.npy'
                        lamk_path = self.landmark_root + real_id + '/' + frame_id
                        landmark=np.load(lamk_path)[0]
                        
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # img, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(img,\
                        #     landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
                        img, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(img,\
                            landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
                        
                        inter_flag = cv2.INTER_CUBIC
                        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
                        img = torch.tensor(img.transpose((2, 0, 1)) / 255., dtype=torch.float32)
                        img = self.norm(img)
                    
                        img = img.unsqueeze(0)
                        video = img if video is None else torch.cat((video, img), dim=0)
                break
        except Exception as e:
            # print(e)
            pass
                    
        return video, target, pattern_lable

    def collate_fn(self, batch):
        video, target, pattern_lable = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        pattern_lable = torch.tensor(pattern_lable).unsqueeze(0)
        return video, target, pattern_lable

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)



from torchvision.transforms import v2
from utils_sbi.library.bi_online_generation import random_get_hull
exist_bi = True
class FFPP_Dataset_V2(Dataset):
    def __init__(self,compress='raw',image_size=224,phase = "train", tamper="all", n_frames=8, patch_num=14):# raw、c23、c40
        super().__init__()
        self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/{compress}/images/'
        self.Deepfakes_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/home/liu/sdb/FFPP/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/home/liu/sdb/FFPP/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Face2Face/{compress}/images/'
        self.patch_num = patch_num
 
        print('start')
        st = time.time()
        # self.original_list = init_fff(phase=phase, dataset_paths=self.original_root)
        # Deepfakes_list = init_fff(phase=phase, dataset_paths=self.Deepfakes_root)
        # NeuralTextures_list = init_fff(phase=phase, dataset_paths=self.NeuralTextures_root)
        # FaseSwap_list = init_fff(phase=phase, dataset_paths=self.FaseSwap_root)
        # Face2Face_list = init_fff(phase=phase, dataset_paths=self.Face2Face_root)
        if phase == 'val' or phase == 'test':
            n_frames = 4
        else:
            # n_frames = 8
            n_frames = n_frames
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=n_frames)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
            # Deepfakes_list[:len(Deepfakes_list)//4] + NeuralTextures_list[len(NeuralTextures_list)//4:len(NeuralTextures_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
            # self.fake_frame_list = Deepfakes_list[:len(Deepfakes_list)//4] + FaseSwap_list[len(FaseSwap_list)//4:len(FaseSwap_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
        elif tamper == 'DF+FS':
            self.fake_frame_list = FaseSwap_list + Deepfakes_list
        elif tamper == 'FF+NT':
            self.fake_frame_list = Face2Face_list + NeuralTextures_list
            
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        # if phase == 'train':
        #     self.fake_frame_list = [self.fake_frame_list[i] for i in range(0, len(self.fake_frame_list), 2)]
        # elif phase == 'val':
        #     self.fake_frame_list = [self.fake_frame_list[i] for i in range(0, len(self.fake_frame_list), 8)]
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))
        # exit(0)
        

        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        self.img_size = image_size
        
        self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.augmentation = self.get_transforms()
        self.phase = phase
        self.tamper = tamper
        
        
    
    def __len__(self):
        return len(self.fake_frame_list)
    
    # #    
    def get_transforms(self):
        return alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RGBShift((-20,20),(-20,20),(-20,20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2), # best
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),   # best
            alb.RandomGridShuffle((self.img_size//self.patch_num, self.img_size//self.patch_num), p=0.15), # best
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)
    
    
    def __getitem__(self, idx):
        # print(idx)
        
        flag = True
        ms_blend = False
        if np.random.rand() < 0.1: # best
        # if np.random.rand() < 0.0:
            ms_blend = True
        while flag:
            try:
                img_path = self.fake_frame_list[idx]
                real_id = img_path.split('/')[-2].split('_')[0]
                
                # real_ID = int(real_id)
                # if real_ID < 250:
                #     real_group_id = [1., 0., 0., 0.]
                # elif 250 <= real_ID < 500:
                #     real_group_id = [0., 1., 0., 0.]
                # elif 500 <= real_ID < 750:
                #     real_group_id = [0., 0., 1., 0.]
                # else:
                #     real_group_id = [0., 0., 0., 1.]
                # real_group_id += [0., 0., 0., 0.]
                
                frame_id = img_path.split('/')[-1].split('.')[0] + '.npy'
                lamk_path = self.landmark_root + real_id + '/' + frame_id
                png_path = img_path.split('images')[-1][1:]
                forgery_type = img_path.split('/')[-5]
                fake_img = cv2.imread(img_path)
                fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                
                real_img_path = self.real_root + real_id + '/' + frame_id.replace(".npy", ".png")
                # real_cluster_label = self.real_dict[real_img_path]
                
                # # remove blend label for real samples
                # true_ind = np.argmax(real_cluster_label)
                # real_cluster_label = [0., 0., 0., 0., 0., 0., 0., 0.]
                # real_cluster_label[true_ind] = 1.
                                
                img = cv2.imread(real_img_path)
                real_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if ms_blend:
                    try: 
                        df_path = self.Deepfakes_root + png_path
                        fs_path = self.FaseSwap_root + png_path 
                        ff_path = self.Face2Face_root + png_path 
                        nt_path = self.NeuralTextures_root + png_path 
                        
                        df_img = cv2.imread(df_path)
                        df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
                        fs_img = cv2.imread(fs_path)
                        fs_img = cv2.cvtColor(fs_img, cv2.COLOR_BGR2RGB)
                        ff_img = cv2.imread(ff_path)
                        ff_img = cv2.cvtColor(ff_img, cv2.COLOR_BGR2RGB)
                        nt_img = cv2.imread(nt_path)
                        nt_img = cv2.cvtColor(nt_img, cv2.COLOR_BGR2RGB)
                        fake_img, _, soft_label_list = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                    except Exception as e:
                        pass
    
                
                    
                # fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending.png", np.uint8(fake_img))
                # FI_map = cv2.cvtColor(np.abs(FI_map), cv2.COLOR_RGB2BGR) * 10
                # FI_map[FI_map>255]=255
                # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending_FT_map.png", np.uint8(FI_map))
                # exit(0)
                
                
                
                landmark=np.load(lamk_path)[0]
                flag = False
            except Exception as e:
                # print(e)
                idx = torch.randint(low=0,high=len(self.fake_frame_list),size=(1,)).item()
        
        landmark=self.reorder_landmark(landmark)
        if self.phase == 'train':
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
            # real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase) # dynamic
        else:
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)

        # if np.random.rand()<0.25:
        #     landmark=landmark[:68]
        # if exist_bi:
        #     mask=random_get_hull(landmark,real_img)[:,:,0]
        # else:
        #     mask=np.zeros_like(real_img[:,:,0])
        #     cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        fake_img = fake_img[y0_new:y1_new,x0_new:x1_new]
        inter_flag = cv2.INTER_CUBIC
        real_img = cv2.resize(real_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        fake_img = cv2.resize(fake_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        
        # if np.random.rand() < 0.1:
        #     fake_img = forgery_intensity_transformation(real_img, fake_img, apply_shape=0., apply_intensity=0.1, apply_frequency=0.)
        # if not mcm:
        #     fake_img = fake_img[y0_new:y1_new,x0_new:x1_new]
        #     inter_flag = cv2.INTER_CUBIC
        #     real_img = cv2.resize(real_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     fake_img = cv2.resize(fake_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     # # mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     # if np.random.rand() < 0.05:
        #     #     mask = augment_mask(mask)
        #     #     mask_blured = get_blend_mask(mask)
        #     #     blend_list=[0.75, 0.8, 0.9,1,1,1]
        #     #     blend_ratio = blend_list[np.random.randint(len(blend_list))]
        #     #     mask_blured*=blend_ratio
        #     #     fake_img=(mask_blured * fake_img + (1 - mask_blured) * real_img)
        # else:
        #     df_img = df_img[y0_new:y1_new,x0_new:x1_new]
        #     fs_img = fs_img[y0_new:y1_new,x0_new:x1_new]
        #     ff_img = ff_img[y0_new:y1_new,x0_new:x1_new]
        #     nt_img = nt_img[y0_new:y1_new,x0_new:x1_new]
        #     inter_flag = cv2.INTER_CUBIC
        #     real_img = cv2.resize(real_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     df_img = cv2.resize(df_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     fs_img = cv2.resize(fs_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     ff_img = cv2.resize(ff_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     nt_img = cv2.resize(nt_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     fake_img, soft_label_list = Multiple_CutMix([df_img, fs_img, ff_img, nt_img])
        #     mask = cv2.resize(mask, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        #     mask = augment_mask(mask)
        #     mask_blured = get_blend_mask(mask)
        #     blend_list=[0.75, 0.8, 0.9,1,1,1]
        #     blend_ratio = blend_list[np.random.randint(len(blend_list))]
        #     mask_blured*=blend_ratio
        #     fake_img=(mask_blured * fake_img + (1 - mask_blured) * real_img)
           
        
        # cv2.imwrite("/home/liu/fcb1/DiM-DiffusionMamba/visualization/real.png", \
        #             cv2.cvtColor(real_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("/home/liu/fcb1/DiM-DiffusionMamba/visualization/fake.png", \
        #     cv2.cvtColor(fake_img.astype("uint8"), cv2.COLOR_RGB2BGR))
        # mask = cv2.absdiff(real_img.astype("float32"), fake_img.astype("float32")) * 2.
        # cv2.imwrite("/home/liu/fcb1/DiM-DiffusionMamba/visualization/mask.png", mask.astype("uint8"))
        # exit(0)
        
        
        transformed=self.augmentation(image=fake_img.astype('uint8'),\
                image1=real_img.astype('uint8'))
        real_img=transformed['image1'].transpose((2, 0, 1)) / 255.
        fake_img=transformed['image'].transpose((2, 0, 1)) / 255. 
        
        
        # if not ms_blend:
        #     if forgery_type == "Deepfakes":
        #         soft_label_list = [1., 0., 0., 0.]
        #     elif forgery_type == "FaceSwap":
        #         soft_label_list = [0., 1., 0., 0.]
        #     elif forgery_type == "Face2Face":
        #         soft_label_list = [0., 0., 1., 0.]
        #     elif forgery_type == "NeuralTextures":
        #         soft_label_list = [0., 0., 0., 1.]

                
        # soft_label_list = [0., 0., 0., 0.] + soft_label_list 

            
        return fake_img, real_img, img_path, real_img_path
        
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark
    
    def collate_fn(self,batch):
        img_f, img_r, img_f_path, img_r_path = zip(*batch)
        data={}
        # if np.random.rand() < 0.01:
        #     class_lab = np.argmax(real_group_id, axis=-1)
        #     if np.random.rand() < 0.5:
        #         img_r, real_group_id = v2.CutMix(alpha=0.1, num_classes=8)(torch.tensor(img_r).float(), torch.tensor(class_lab))
        #     else:
        #         img_r, real_group_id = v2.MixUp(alpha=0.1,num_classes=8)(torch.tensor(img_r).float(), torch.tensor(class_lab))
        #     data["real_group_id"]=real_group_id
        #     data['img']=torch.cat([img_r, torch.tensor(img_f).float()],0)
        # else:
        #     data["real_group_id"]=torch.tensor(real_group_id)
        #     data['img']=torch.cat([torch.tensor(img_r).float(), torch.tensor(img_f).float()],0)
        data['img'] = torch.cat([torch.tensor(img_r).float(), torch.tensor(img_f).float()],0)
        data['label'] = torch.tensor([0]*len(img_r)+[1]*len(img_f))
        data["path"] = img_r_path + img_f_path
        
        
        return data

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        
        
from skimage.transform import PiecewiseAffineTransform, warp
def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)   # 生成坐标
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)        # 得到对应的坐标矩阵，即两个矩阵中分别保存x轴坐标与y轴坐标

    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    if np.random.rand() < 0.5:
        warped = warp(image, trans, order=3).astype(np.float32)
        warped = warp(warped, trans.inverse, order=3).astype(np.float32)
    else:
        warped = warp(image, trans, order=1).astype(np.float32)
        warped = warp(warped, trans.inverse, order=1).astype(np.float32)
    return warped



#v1 best
def forgery_intensity_transformation(real, fake, apply_shape=0.1, apply_intensity=0.1, apply_frequency=0.2):
    
    real = real.astype(np.float32)
    fake = fake.astype(np.float32)
    img_size = real.shape[0]
    residual_map = real - fake
    warped = residual_map
    
    p = np.random.rand()
    if p <= apply_shape:
        grid_size_list = [2, 3, 4] # best
        grid_size = np.random.choice(grid_size_list)
        if grid_size != 0:
            lamk, lamk_deformed = random_deform((img_size, img_size), grid_size, grid_size)
            warped = piecewise_affine_transform(warped, lamk, lamk_deformed)
        else:
            warped = residual_map
    
    p = np.random.rand()
    if p <= apply_intensity:
        weight_of_residual_list = [0.7, 0.8, 0.9] # best 
        w = np.random.choice(weight_of_residual_list)
        warped = warped * w
    


    p = np.random.rand()
    if p < apply_frequency:
        for i in range(3):
            cA, (cH, cV, cD) = pywt.dwt2(warped[:,:,i], 'haar')
            cA = cv2.GaussianBlur(cA, ksize=(5, 5), sigmaX=0)
            warped = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        # warped = cv2.GaussianBlur(warped, ksize=(5, 5), sigmaX=0)

    fake_warped = real - warped
    fake_warped[fake_warped > 255] = 255
    fake_warped[fake_warped < 0] = 0
    
    return fake_warped
       
       
if __name__ == "__main__":
    seed = 11
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print('Random seed :', seed)
    # train_dataset = FFPP_Dataset(phase='train',image_size=224,compress='c23',tamper='FF')
    train_dataset = FFPP_Dataset_V2(phase='train',image_size=224,compress='c23',tamper='all')
    # train_dataset = VideoFFPP_Dataset(phase='test',image_size=224,compress='c23',tamper='all')
    # train_dataset.__getitem__(2)
    batch_size_sbi = 32
    train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = 4,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn
                                            )
    print(len(train_set))
    # exit(0)
    for step_id, data in enumerate(train_set):
        img = data['img']
        # print(img.shape)
        # label = data['label']
        # print(label)
        # s_label = data['soft_labels']
        # print(s_label)
    

    