# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:53:29 2023

@author: user
"""

import sys
import glob
import re
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

device = torch.device('cuda')
dtype = torch.float32

class AEData(Dataset):
    def __init__(self,x_data, device = device):
        self.x_data = torch.FloatTensor(x_data)
        self.len = self.x_data.shape[0]
        self.device = device
        self.cache = {}
        self.cache['device'] = 'cpu'
        self.cache['Std'] = 'Org'
    
    def __call__(self):
        return self.x_data
    
    def __getitem__(self, index):
        return self.x_data[index]
    
    def __len__(self):
        return self.len
    
    def cuda(self):
        self.x_data = self.x_data.to(device = device)
        self.cache['device'] = 'cuda'
        
    def cpu(self):
        self.x_data = self.x_data.to('cpu')
        self.cache['device'] = 'cpu'
    
    def std(self):
        try:
            self.clone = self.x_data.clone().to('cpu')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.x_data = (self.x_data - self.x_data.mean(0,keepdims=True))/self.x_data.std(0,keepdims=True)
            
            self.x_data = torch.nan_to_num(self.x_data)
            self.cache['std'] = 'std'
        except:
            pass
            
    def org(self):
        try:
            self.x_data = self.clone.to(self.cache['device'])
            del self.clone
            self.cache['std'] = 'org'
        except:
            pass
    
    def cache(self):
        print(self.cache)

def TrainDataLoader():

    data_paths =  glob.glob('Data\\Train\\*.csv')
    n = True
    
    print('\n  ###   Train Data Load   ###')
    for path in tqdm(data_paths):
        if n:
            Train_Data = np.loadtxt(path,delimiter=',',dtype=str)[1:].astype(float)
            n = not n
        else:
            Train_Data = np.r_[Train_Data,np.loadtxt(path,delimiter=',',dtype=str)[1:].astype(float)]
            
    size = sys.getsizeof(Train_Data)
    size = str(round(size/1024**3,1)) + 'GB' if size/1024**3>1 else str(round(size/1024**2,1)) + 'MB'
    print(f'  Total {len(Train_Data)} rows and {size} size')
    
    Train_Data[Train_Data<0] = 0
    return AEData(Train_Data)


def TestDataLoader():
    Test_Data = {}
    Dict_Var = [d.split('\\')[-1] for d in glob.glob('Data/Test/*')]

    print('\n  ###   Test Data Load   ###')
    
    N = 0
    
    for v in tqdm(Dict_Var):
        
        data_paths = glob.glob(f'Data\\Test\\{v}\\1\\*.xlsx')
        data_names = [p.split('\\')[-1] for p in data_paths]
    
        for n, name in enumerate(data_names):
        
            condition = tuple([int(n) for n in re.findall('\d+',name)])
            temp_data = pd.read_excel(data_paths[n], 'Sheet1').to_numpy()
            
            N += len(temp_data) 
        
            if condition in list(Test_Data.keys()):
                Test_Data[condition] = np.r_[Test_Data[condition],temp_data]
            
            else:
                Test_Data[condition] = temp_data
                
    size = sys.getsizeof(Test_Data)
    size = str(round(size/1024**3,1)) + 'GB' if size/1024**3>1 else str(round(size/1024**2,1)) + 'MB'
    print(f'  Total {N} rows and {size} size')
    
    print('\n  ###   Shapes in X, Y   ###')
    n = True
    
    for k,v in Test_Data.items():
        N = len(v)
        temp_y = np.repeat(np.array(k)[np.newaxis,:],N,axis=0)
        if n:
            X = v
            Y = temp_y
            n = not n
        else:
            X = np.r_[X,v]
            Y = np.r_[Y,temp_y]
    X[X<0] = 0
    return AEData(X),Y


