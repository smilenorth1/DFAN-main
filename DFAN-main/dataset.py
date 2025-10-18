import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import cv2
import matplotlib.pyplot as plt

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    # img = Image.open(filepath).convert('RGB') #oringinal method
    img=cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img=np.array(img,dtype=np.float32)
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.shape
    new_size_in = tuple([int(size_in[1] * scale),int(size_in[0]*scale)]) #must be in the order of size_in[1],size_in[0]
    img_in = cv2.resize(img_in,new_size_in, interpolation=cv2.INTER_CUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, ix=-1, iy=-1):
    (ih, iw,_) = img_in.shape
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    img_in = img_in[ix:ix + ip, iy:iy + ip,:]
    img_tar = img_tar[ix:ix + ip, iy:iy + ip,:]
    img_bic = img_bic[ix:ix + ip, iy:iy + ip,:]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip}
    return img_in, img_tar, img_bic, info_patch

# def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
#     info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
#     if random.random() < 0.5 and flip_h:
#         img_in = ImageOps.flip(img_in)
#         img_tar = ImageOps.flip(img_tar)
#         img_bic = ImageOps.flip(img_bic)
#         info_aug['flip_h'] = True

#     if rot:
#         if random.random() < 0.5:
#             img_in = ImageOps.mirror(img_in)
#             img_tar = ImageOps.mirror(img_tar)
#             img_bic = ImageOps.mirror(img_bic)
#             info_aug['flip_v'] = True
#         if random.random() < 0.5:
#             img_in = img_in.rotate(180)
#             img_tar = img_tar.rotate(180)
#             img_bic = img_bic.rotate(180)
#             info_aug['trans'] = True
            
#     return img_in, img_tar, img_bic, info_aug

def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :].copy()
        if vflip:
            img = img[::-1, :, :].copy()
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, bit_gap, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.bit_gap = bit_gap
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        quant_bin = 2.0 ** (self.bit_gap)
        input = np.floor(target/quant_bin)*quant_bin
        ZP = input
        input, target, ZP, _ = get_patch(input,target,ZP,self.patch_size)
        if self.data_augmentation:
            input, target, ZP = augment([input, target, ZP])
        if self.transform:
            input = self.transform(input)
            ZP = self.transform(ZP)
            target = self.transform(target) 
        # input1=input.permute(1,2,0)
        # ZP1=ZP.permute(1,2,0)
        # target1=target.permute(1,2,0)
        # plt.figure()
        # plt.imshow((target1)/65535.0)
        # plt.title('GT')
        # plt.figure()
        # plt.imshow((input1)/65535.0)
        # plt.title('LR')
        # plt.figure()
        # plt.imshow((ZP1)/65535.0)
        # plt.title('ZP')
        # plt.show()
        return input/65535.0, target/65535.0, ZP/65535.0
        

    def __len__(self):
        return len(self.image_filenames)

def get_patch_test(img_in, img_tar, img_bic, scale=1, patch_size_factor=8, multi_scale=False):
    ih, iw = img_in.shape[:2]
    p = scale if multi_scale else 1

    ix = (iw // patch_size_factor) * patch_size_factor
    iy = (ih // patch_size_factor) * patch_size_factor
    tx, ty = p * ix, p * iy

    img_in = img_in[0:iy, 0:ix, :]
    img_tar = img_tar[0:ty, 0:tx, :]
    img_bic = img_bic[0:ty, 0:tx, :]
    return img_in, img_tar, img_bic

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, hr_dir, bit_gap, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.bit_gap = bit_gap
        self.transform = transform
        # print(self.image_filenames[20])
        # print(self.image_filenames[21])
        # print(self.image_filenames[28])

    def __getitem__(self, index):
        quant_bin = 2.0 ** (self.bit_gap)
        HBD = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        input=np.floor(HBD/quant_bin)*quant_bin   # np.floor(HBD/4096.0)：通用比特截断，将16比特变成4比特
        ZP = input
        # input, HBD, ZP = get_patch_test(input, HBD, ZP, 1)
        
        if self.transform:
            # print("%%%%%%%%%%%%%%%%%%")
            HBD=self.transform(HBD)
            input = self.transform(input)
            ZP = self.transform(ZP)
        # print(HBD.shape)
        # print(input.shape)
        # print(HBD)
        # print(input)    
        return input/65535, ZP/65535, HBD/65535, file

      
    def __len__(self):
        return len(self.image_filenames)
