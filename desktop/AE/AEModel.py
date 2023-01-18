# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:03:42 2022

@author: user
"""
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


device = torch.device('cuda')
dtype = torch.float32

class FCCAutoencoder(nn.Module):
    def __init__(self, D, Z, num_layer, p_Dropout = 0.8, normalization = None, 
                 weight_scale = 1e-2,seed = None , dtype = dtype):
        
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        
        self.num_layer = num_layer
        self.normalization = normalization
        self.use_dropout = p_Dropout != 1
        self.dtype = dtype
        
        # encode
        self.hidden = {}
        self.params = nn.ParameterDict()
        
        w1 = D
        
        for n in range(self.num_layer):
            
            if n == self.num_layer-1:
                w2 = Z
                
            else:
                w2 = w1
            
            self.hidden['E%d'%(n+1)] = nn.Linear(w1, w2, bias=True, device=device,dtype=dtype)
            nn.init.kaiming_normal_(self.hidden['E%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['E%d'%(n+1)].bias)
            
            self.params['E%d_weight'%(n+1)] = self.hidden['E%d'%(n+1)].weight
            self.params['E%d_bias'%(n+1)] = self.hidden['E%d'%(n+1)].bias
            
            if normalization == 'batchnorm':
                self.hidden['BN_E%d'%(n+1)] = nn.BatchNorm1d(w2, device=device)
                self.params['BNE%d_weight'%(n+1)] = self.hidden['BN_E%d'%(n+1)].weight
                self.params['BNE%d_bias'%(n+1)] = self.hidden['BN_E%d'%(n+1)].bias
            if normalization == 'layernorm':
                self.hidden['LN_E%d'%(n+1)] = nn.LayerNorm(w2, device=device)
                self.params['LNE%d_weight'%(n+1)] = self.hidden['LN_E%d'%(n+1)].weight
                self.params['LNE%d_bias'%(n+1)] = self.hidden['LN_E%d'%(n+1)].bias

            
        # decode
        
        for n in range(self.num_layer):
            
            if n == 0:
                w1 = Z
                w2 = D
            else:
                w1 = w2
                
            self.hidden['D%d'%(n+1)] = nn.Linear(w1, w2, device=device,dtype=dtype)
            nn.init.kaiming_normal_(self.hidden['D%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['D%d'%(n+1)].bias)
            
            self.params['D%d_weight'%(n+1)] = self.hidden['D%d'%(n+1)].weight
            self.params['D%d_bias'%(n+1)] = self.hidden['D%d'%(n+1)].bias
            
            if normalization == 'batchnorm':
                self.hidden['BN_D%d'%(n+1)] = nn.BatchNorm1d(w2, device=device)
                self.params['BND%d_weight'%(n+1)] = self.hidden['BN_D%d'%(n+1)].weight
                self.params['BND%d_bias'%(n+1)] = self.hidden['BN_D%d'%(n+1)].bias
            if normalization == 'layernorm':
                self.hidden['LN_D%d'%(n+1)] = nn.LayerNorm(w2, device=device)
                self.params['LND%d_weight'%(n+1)] = self.hidden['LN_D%d'%(n+1)].weight
                self.params['LND%d_bias'%(n+1)] = self.hidden['LN_D%d'%(n+1)].bias
            
            w1 = w2 
        
        if self.use_dropout:
            self.DO = nn.Dropout(p_Dropout)
            
    def forward(self,X):
            

        if self.normalization != None:
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_E%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_E%d'%(n+1)].training = self.training
            
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_D%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_D%d'%(n+1)].training = self.training
                    
        temp_h = X

        #encode
        for n in range(self.num_layer):
            temp_h = self.hidden['E%d'%(n+1)](temp_h)
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_E%d'%(n+1)](temp_h)
            elif self.normalization == 'layernorm':
                temp_h = self.hidden['LN_E%d'%(n+1)](temp_h)
            temp_h = F.relu(temp_h)
            temp_h = self.DO(temp_h)

        #decode
        for n in range(self.num_layer):
            temp_h = self.hidden['D%d'%(n+1)](temp_h)
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_D%d'%(n+1)](temp_h)
            if self.normalization == 'layernorm':
                temp_h = self.hidden['LN_D%d'%(n+1)](temp_h)
            temp_h = F.relu(temp_h)
            temp_h = self.DO(temp_h)
            
        return temp_h
        
    def encode(self,x):
        
        if self.normalization != None:
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_E%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_E%d'%(n+1)].training = self.training
            
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_D%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_D%d'%(n+1)].training = self.training
        temp_h = x
        
        #encode
        for n in range(self.num_layer):
            temp_h = self.hidden['E%d'%(n+1)](temp_h)
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_E%d'%(n+1)](temp_h)
            if self.normalization == 'layernorm':
                temp_h = self.hidden['LN_E%d'%(n+1)](temp_h)
            temp_h = F.relu(temp_h)
        
        return temp_h


class LOGAutoencoder(nn.Module):
    def __init__(self,D,Z,num_layer, p_Dropout = 0.8, normalization = None, 
                 weight_scale = 1e-2,seed = None , dtype = dtype):
        
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        
        self.num_layer = num_layer
        self.normalization = normalization
        self.use_dropout = p_Dropout != 1
        self.dtype = dtype
        
        # encode
        self.hidden = {}
        self.params = nn.ParameterDict()
        
        
        w1 = D
        
        for n in range(self.num_layer):
            
            if n == self.num_layer-1:
                w2 = Z
                
            else:
                w2 = w1
            
            self.hidden['E1%d'%(n+1)] = nn.Linear(w1, w2, bias=True, device=device,dtype=dtype)
            self.hidden['E2%d'%(n+1)] = nn.Linear(w2, w2, bias=True, device=device,dtype=dtype)
            nn.init.kaiming_normal_(self.hidden['E1%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['E1%d'%(n+1)].bias)
            nn.init.kaiming_normal_(self.hidden['E2%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['E2%d'%(n+1)].bias)
            
            self.params['E1%d_weight'%(n+1)] = self.hidden['E1%d'%(n+1)].weight
            self.params['E2%d_weight'%(n+1)] = self.hidden['E2%d'%(n+1)].weight
            self.params['E1%d_bias'%(n+1)] = self.hidden['E1%d'%(n+1)].bias
            self.params['E2%d_bias'%(n+1)] = self.hidden['E2%d'%(n+1)].bias
            
            if normalization == 'batchnorm':
                self.hidden['BN_E%d'%(n+1)] = nn.BatchNorm1d(w2, device=device)
                self.params['BNE%d_weight'%(n+1)] = self.hidden['BN_E%d'%(n+1)].weight
                self.params['BNE%d_bias'%(n+1)] = self.hidden['BN_E%d'%(n+1)].bias
            if normalization == 'layernorm':
                self.hidden['LN_E%d'%(n+1)] = nn.LayerNorm(w2, device=device)
                self.params['LNE%d_weight'%(n+1)] = self.hidden['LN_E%d'%(n+1)].weight
                self.params['LNE%d_bias'%(n+1)] = self.hidden['LN_E%d'%(n+1)].bias
            
            w1 = w2
            
        # decode
        
        for n in range(self.num_layer):
            
            if n == 0:
                w1 = Z
                w2 = D
            else:
                w1 = w2
                
            self.hidden['D1%d'%(n+1)] = nn.Linear(w1, w2, device=device,dtype=dtype)
            self.hidden['D2%d'%(n+1)] = nn.Linear(w2, w2, device=device,dtype=dtype)
            nn.init.kaiming_normal_(self.hidden['D1%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['D1%d'%(n+1)].bias)
            nn.init.kaiming_normal_(self.hidden['D2%d'%(n+1)].weight)
            nn.init.normal_(self.hidden['D2%d'%(n+1)].bias)
            
            self.params['D1%d_weight'%(n+1)] = self.hidden['D1%d'%(n+1)].weight
            self.params['D2%d_weight'%(n+1)] = self.hidden['D2%d'%(n+1)].weight
            self.params['D1%d_bias'%(n+1)] = self.hidden['D1%d'%(n+1)].bias
            self.params['D2%d_bias'%(n+1)] = self.hidden['D2%d'%(n+1)].bias
            
            if normalization == 'batchnorm':
                self.hidden['BN_D%d'%(n+1)] = nn.BatchNorm1d(w2, device=device)
                self.params['BND%d_weight'%(n+1)] = self.hidden['BN_D%d'%(n+1)].weight
                self.params['BND%d_bias'%(n+1)] = self.hidden['BN_D%d'%(n+1)].bias
            if normalization == 'layernorm':
                self.hidden['LN_D%d'%(n+1)] = nn.LayerNorm(w2, device=device)
                self.params['LND%d_weight'%(n+1)] = self.hidden['LN_D%d'%(n+1)].weight
                self.params['LND%d_bias'%(n+1)] = self.hidden['LN_D%d'%(n+1)].bias
            
            w1 = w2 
            
        self.DO = nn.Dropout(p_Dropout)
            
    def forward(self,X):
        
        if (torch.is_tensor(X)):
            pass
        
        else:
            X = torch.tensor(X,device=device,dtype=dtype)
        
        if self.normalization != None:
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_E%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_E%d'%(n+1)].training = self.training
            
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_D%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_D%d'%(n+1)].training = self.training
        temp_h = X
        
        #encode
        for n in range(self.num_layer):
            temp_h = torch.log(temp_h+1e-5)
            temp_h = self.DO(temp_h)
            temp_h = self.hidden['E1%d'%(n+1)](temp_h)
            temp_h = torch.exp(temp_h)
            temp_h = self.hidden['E2%d'%(n+1)](temp_h)
            temp_h = self.DO(temp_h)
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_E%d'%(n+1)](temp_h)
                
            if self.normalization == 'layernorm':
                temp_h = self.hidden['LN_E%d'%(n+1)](temp_h)
            

        #decode
        for n in range(self.num_layer):
            temp_h = self.hidden['D1%d'%(n+1)](temp_h)
            temp_h = torch.log(temp_h+1e-5)
            temp_h = self.DO(temp_h)
            temp_h = self.hidden['D2%d'%(n+1)](temp_h)
            temp_h = temp_h/torch.norm(temp_h,dim=1,keepdim=True)
            temp_h = torch.exp(temp_h)
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_E%d'%(n+1)](temp_h)
            if self.normalization == 'layernorm':
                temp_h = self.hidden['LN_E%d'%(n+1)](temp_h)
            
        return temp_h

        
    def encode(self,x):
        
        if self.normalization != None:
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_E%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_E%d'%(n+1)].training = self.training
            
            for n in range(self.num_layer):
                if self.normalization == 'batchnorm':
                    self.hidden['BN_D%d'%(n+1)].training = self.training
                elif self.normalization == 'layernorm':
                    self.hidden['LN_D%d'%(n+1)].training = self.training
                    
        if torch.is_tensor(x):
            pass
        else:
            x = torch.tensor(x,device=device,dtype=dtype)
            
        temp_h = x
        
        #encode
        for n in range(self.num_layer):
            temp_h = torch.log(temp_h+1e-5)
            temp_h = self.hidden['E1%d'%(n+1)](temp_h)
            temp_h = torch.exp(temp_h)
            temp_h = self.hidden['E2%d'%(n+1)](temp_h)
            
            if self.normalization == 'batchnorm':
                temp_h = self.hidden['BN_E%d'%(n+1)](temp_h)
            if self.normalization == 'layernorm':
                temp_h = self.hidden['LN_E%d'%(n+1)](temp_h)
                
        return temp_h
    

    
def train(Model,X,num_Z, epoch, title = None, normalization = None,  num_layer = 10,batch_size = 32, orthogonal = False):

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
    best_model = None
    best_loss = 2
    
    # criterion_MSE = nn.MSELoss()
    
    date = time.strftime('%Y-%m-%d',time.localtime())

    for f, (idx_t, idx_v) in enumerate(kfold.split(X)):
        writer = SummaryWriter(log_dir=f'logs/Orthogonal_{orthogonal}/Nor_{normalization}/{title}_fold{f}')
        Title = f'{date} {title} loss'
        train_idx = torch.utils.data.SubsetRandomSampler(idx_t)
        val_idx = torch.utils.data.SubsetRandomSampler(idx_v)
        
        trainloader = DataLoader(X, batch_size=batch_size, sampler=train_idx)
        valloader = DataLoader(X, batch_size=batch_size, sampler=val_idx)
        
        model = Model(3648, num_Z, num_layer = num_layer, normalization = normalization )
        model.train()
        
        if orthogonal:
            criterion_MSE = nn.MSELoss()
            
        criterion_Cos = nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        model.train()
        
        m = 0
        for e in range(int(epoch)):
            
            model.train()
            for x in trainloader:
                m += 1
                scores = model(x)

                loss_cos = criterion_Cos(scores,x,torch.tensor([1],device=device))
                    
                optimizer.zero_grad()
                loss = loss_cos
                
                if orthogonal:
                    I = torch.eye(num_Z,device=device)
                    loss *= criterion_MSE(torch.cov(model.encode(x).T),I)
                    
                loss.backward()
                optimizer.step()
                writer.add_scalar(Title + '/Train_loss', loss, m)


        # loss_mse_val = criterion_MSE(Model(x_val),x_val)
            model.eval()
            loss_cos_val = 0
            for n,x in enumerate(valloader):
                loss_cos_val += criterion_Cos(model(x),x,torch.tensor([1],device=device)).item()
            loss_val = loss_cos_val/(n+1)
            writer.add_scalar(Title + '/Val_loss', loss_val, m)
            scheduler.step(loss_val)
            writer.add_scalar(Title + '/lr',optimizer.param_groups[0]['lr'],m)
        writer.close()

        if best_loss > loss_val:
            best_model = model
            best_loss = loss_val
            
        else:
            pass
        
    return best_model

