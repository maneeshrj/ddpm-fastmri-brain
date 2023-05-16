#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset classes for loading FastMRI brain data.

@author: mrjohn
"""
    
import pickle
import numpy as np
import yaml
import argparse
import os
#import h5py as h5
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

from mri import sense

# FastMRI brain subjects
# 108 FLAIR, 104 T1, 113 T1POST, 115 T2

#%%        

def get_labels_key(path):
    subdirs = sorted(os.listdir(path))
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    return labels_key

def preload(path, start_sub=0, num_sub_per_type=2, acc=4.0, num_sl=10):
    subdirs = sorted(os.listdir(path))
    train_ksp, train_csm, labels = None, None, None
    for i, subdir in enumerate(subdirs):
        fnames = [filename for filename in sorted(os.listdir(path+subdir)) if filename.endswith('.pickle')]
        print(subdir, '- loading', num_sub_per_type, 'of', len(fnames), 'subjects')
        
        subpath = os.path.join(path, subdir)
        train_fnames = fnames[start_sub:start_sub+num_sub_per_type]
        
        for j, train_fname in enumerate(train_fnames):
            with open(os.path.join(subpath, train_fname), 'rb') as f:
                ksp, csm = pickle.load(f)
                ksp, csm = ksp[:num_sl], csm[:num_sl]
                if i==0 and j==0:
                    train_ksp = torch.tensor(ksp)
                    train_csm = torch.tensor(csm)
                    labels = torch.ones(ksp.shape[0],)*i
                else:
                    train_ksp = torch.cat((train_ksp, torch.tensor(ksp)))
                    train_csm = torch.cat((train_csm, torch.tensor(csm)))
                    labels = torch.cat((labels, torch.ones(ksp.shape[0],)*i))
                print('ksp:', ksp.shape, '\tcsm:', csm.shape)
        
    # print('ksp:', train_ksp.shape, '\ncsm:', train_csm.shape, '\nlabels:', labels.shape,)
    
    if acc == 0:
        mask = torch.ones_like(train_ksp)
    elif acc != None:
        mask_filename = f'poisson_mask_2d_acc{acc:.1f}_320by320.npy'
        # mask = np.load(mask_filename)
        # mask = torch.tensor(mask)
        mask = np.load(mask_filename).astype(np.complex64)  
        mask = torch.tensor(np.tile(mask, [train_ksp.shape[0],train_ksp.shape[1],1,1]))
        # print("mask:", mask.shape)
    else:
        mask = None
    
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    print(f"Loaded dataset of {train_ksp.shape[0]} slices\n")
    
    return train_ksp, train_csm, mask, labels.long(), labels_key

def preprocess(ksp, csm, mask):    
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    us_ksp = ksp * mask
    
    return org, us_ksp.type(torch.complex64), csm.type(torch.complex64), mask

def preprocess_imgs_complex(ksp, csm):
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    
    # print('min/max:', org.abs().min(), '/', org.abs().max()) 
    
    return org

def preprocess_imgs_mag(ksp, csm):
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True).abs()
    for i in range(org.shape[0]):
        org[i] = org[i]/org[i].max()
    print('min/max:', org.abs().min(), '/', org.abs().max()) 
    
    return org

               
#%% Returns images, undersampled kspace, coil sensitivity maps, & masks
class DataGenBrain(Dataset):
    def __init__(self, start_sub=0, num_sub=2, device=None, acc=4.0, data_path='/Shared/lss_jcb/aniket/FastMRI brain data/'):
        self.path = data_path
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        self.acc = acc
        self.ksp, self.csm, self.msk, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, acc=acc)
        self.org, self.us_ksp, self.csm, self.msk = preprocess(self.ksp, self.csm, self.msk)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        return self.org[i:i+1].to(self.device), self.us_ksp[i:i+1].to(self.device), self.csm[i:i+1].to(self.device), self.msk[i:i+1].to(self.device), self.labels[i:i+1].to(self.device)
    
    def get_noisy(self, i, noise_eps=0.):
        us_ksp = self.us_ksp[i:i+1] 
        msk = self.msk[i:i+1]
        scale = 1/torch.sqrt(torch.tensor(2.))
        us_ksp = us_ksp + msk*(torch.randn(us_ksp.shape)+1j*torch.randn(us_ksp.shape))*scale*noise_eps
        
        return self.org[i:i+1].to(self.device), us_ksp.to(self.device), self.csm[i:i+1].to(self.device), msk.to(self.device), self.labels[i:i+1].to(self.device)
    
    
#%% Returns full-resolution complex images
class DataGenImagesOnly(Dataset):
    def __init__(self, start_sub=0, num_sub=2, device=None, data_path='/Shared/lss_jcb/aniket/FastMRI brain data/'):
        self.path = data_path
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        ksp, csm, _, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, None)
        self.org = preprocess_imgs_complex(ksp, csm)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        return self.org[i:i+1].to(self.device), self.labels[i:i+1].to(self.device)#, self.labels_key[self.labels[i].item()]

    
#%% Returns complex or magnitude images resized to a lower resolution
class DataGenImagesDownsampled(Dataset):
    def __init__(self, start_sub=0, num_sub=2, device=None, res=0, complex_in=False, data_path='/Shared/lss_jcb/aniket/FastMRI brain data/', num_sl=None):
        self.path = data_path
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        ksp, csm, _, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, acc=None, num_sl=num_sl)
        if complex_in:
            self.org = preprocess_imgs_complex(ksp, csm)
            if res > 0:
                self.org = self.downsample_2d(self.org.real, res) + self.downsample_2d(self.org.imag, res)*1j
        else:
            self.org = preprocess_imgs_mag(ksp, csm).float()
            if res > 0:
                self.org = self.downsample_2d(self.org, res)
            
        print('Resized org:', self.org.shape)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        # return self.org[i:i+1].to(self.device), self.labels[i:i+1].to(self.device)#, self.labels_key[self.labels[i].item()]
        return self.org[i], self.labels[i]

    def downsample_2d(self, X, sz):
        """
        Downsamples a stack of square images.

        Args:
            X: a stack of images (batch, channels, ny, ny).
            sz: the desired size of images.

        Returns:
            The downsampled images, a tensor of shape (batch, channel, sz, sz)
        """
        kernel = torch.tensor([[.25, .5, .25], 
                               [.5, 1, .5], 
                               [.25, .5, .25]], device=X.device).reshape(1, 1, 3, 3)
        kernel = kernel.repeat((X.shape[1], 1, 1, 1))
        while sz < X.shape[-1] / 2:
            # Downsample by a factor 2 with smoothing
            mask = torch.ones(1, *X.shape[1:])
            mask = F.conv2d(mask, kernel, groups=X.shape[1], stride=2, padding=1)
            X = F.conv2d(X.float(), kernel, groups=X.shape[1], stride=2, padding=1)

            # Normalize the edges and corners.
            X = X = X / mask

        return F.interpolate(X, size=sz, mode='bilinear')        