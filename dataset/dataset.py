import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import copy
import random
from typing import Sequence

class dataset_loader(data.Dataset):
    def __init__(self, opt, status='train'):
        self.opt = opt
        self.crop_size = opt.crop_size
        self.status = status           
        if self.status == 'train':      
            self.clean_dir_path = os.path.join(opt.dataset_path,'train/clean')
            self.noise_dir_path = os.path.join(opt.dataset_path,'train/noise_'+opt.noise)
        elif self.status == 'val':      
            self.clean_dir_path = os.path.join(opt.dataset_path,'val4/clean')
            self.noise_dir_path = os.path.join(opt.dataset_path,'val4/noise_'+opt.noise)
        elif self.status == 'test':      
            # self.clean_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/CHRIS/CHRIS_dataset3_400/test'
            # self.noise_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/CHRIS/CHRIS_dataset3_400/test'              
            # self.clean_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/NIR/NIR_outdoor1/test/noise'
            # self.noise_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/NIR/NIR_outdoor1/test/noise'   
            # self.clean_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/BU_TIV_dataset2/test'
            # self.noise_dir_path = '/opt/data/private/qc/02_self_denoise/00_dataset/BU_TIV_dataset2/test'                    
            self.clean_dir_path = os.path.join(opt.dataset_path,'val4/clean')
            self.noise_dir_path = os.path.join(opt.dataset_path,'val4/noise_'+opt.noise)                                           
        self.image_filenames = [x for x in os.listdir(self.clean_dir_path) if is_image_file(x)]
        self.image_filenames.sort(key=lambda x:int(x[0:-4]))
        transform_train_list = [transforms.ToTensor()]
        if 'crop' in opt.data_preprocess:
            transform_train_list.append(transforms.RandomCrop(self.crop_size))
        if 'Hflip' in opt.data_preprocess:
            transform_train_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if 'Vflip' in opt.data_preprocess:
            transform_train_list.append(transforms.RandomVerticalFlip(p=0.5))            
        if 'rotate' in opt.data_preprocess:
            transform_train_list.append(RotateTransform([0, 180])) 
        self.transform_train = transforms.Compose(transform_train_list)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        if self.opt.input_ch == 1:
            clean = Image.open(os.path.join(self.clean_dir_path,self.image_filenames[index])).convert('L')
            noise = Image.open(os.path.join(self.noise_dir_path,self.image_filenames[index])).convert('L')           
            clean = np.expand_dims(clean,axis=-1)
            noise = np.expand_dims(noise,axis=-1)
        elif self.opt.input_ch == 3:
            clean = Image.open(os.path.join(self.clean_dir_path,self.image_filenames[index])).convert('RGB')          
            noise = Image.open(os.path.join(self.noise_dir_path,self.image_filenames[index])).convert('RGB')  
        else:
            print('data load error')
        if self.status == 'train':
            noise = self.transform_train(noise)
            return noise
        elif self.status == 'val':
            noise = self.to_tensor(noise) 
            clean = self.to_tensor(clean)         
            return clean,noise
        elif self.status == 'test':
            noise = self.to_tensor(noise) 
            clean = self.to_tensor(clean) 
            return clean,noise,img_name

    def __len__(self):
        if self.status == 'train':
            return len(self.image_filenames) 
        elif self.status == 'val':
            return 1
        elif self.status == 'test':
            return len(self.image_filenames)     

class RotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']) 