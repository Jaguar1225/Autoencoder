# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 09:44:06 2022

@author: user
"""

#%%

'''

Module import

'''
from AEModel import FCCAutoencoder, LOGAutoencoder, train
from Data_loader import TrainDataLoader, TestDataLoader


from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

import time
import matplotlib.pyplot as plt


device = torch.device('cuda')
dtype = torch.float32


#%%
'''

Train Data Load

'''
AEData_train = TrainDataLoader()
AEData_test, Y_test = TestDataLoader()


#%%
'''

FCL AE


'''


m = FCCAutoencoder

Pre = [False,True]
norm = [None,'batchnorm','layernorm']
Nor = ['None','BN','LN']
num_layer = [n for n in range(1,6)]
num_Z = [1824,912,456,228,114,57,28,14,7,4,2]

num_Var = len(Pre)*len(norm)*len(num_layer)*len(num_Z)



criterion_Cos = nn.CosineEmbeddingLoss()
loss_log = {}

idx = 1
total_time = 0 

for pre in Pre:
    if pre:
       AEData_train.std()
       AEData_test.std()
    for N,n in enumerate(norm):
        for l in tqdm(num_layer):
            for z in num_Z:
                time_f = time.time()
                torch.cuda.empty_cache()
                plt.clf()
                AEData_train.cuda()
                title = f'FCL_AE_Pre{pre}_{Nor[N]}_FCC_layer_num{l}_Z{z}'
                try:
                    model = train(FCCAutoencoder, AEData_train, z,100, title = title, num_layer = l,normalization = n,batch_size=2048)
                except:
                    model = train(FCCAutoencoder, AEData_train, z,100, title = title, num_layer = l,normalization = n,batch_size=256)
                
                AEData_train.cpu()
                AEData_test.cuda() 
                
                model.eval()
                
                loss_log[pre,Nor[N],l,z] = criterion_Cos(model(AEData_test()),AEData_test(),torch.tensor([1],device=device)).item()
                fig, axs = plt.subplots(2)
                fig.suptitle(title)
                axs[0].plot(AEData_test()[0].to('cpu'))
                axs[0].grid()
                Decode = model(AEData_test()[0:1]).detach().to('cpu')
                axs[1].plot(Decode.T)
                axs[1].grid()
                plt.show()
                np.savetxt(f'Result/FCL_AE/Decoding/{title}.csv',Decode,delimiter=',')
                del Decode
                fig.savefig(f'Fig/FCL_AE/{title}_Decoding')
                plt.clf()

                
                fig, axs = plt.subplots(2)
                fig.suptitle(title)
                axs[0].plot(Y_test)
                axs[0].grid()
                Encode = model.encode(AEData_test()).detach().to('cpu')
                axs[1].plot(Encode)
                axs[1].grid()
                plt.show()
                AEData_test().cpu()
                np.savetxt(f'Result/FCL_AE/Encoding/{title}.csv',Encode,delimiter=',')
                del Encode
                fig.savefig(f'Fig/FCL_AE/{title}_Encoding',dpi=300)
                plt.clf()
                
                torch.save(model,f'Model/FCL_AE/{title}.pt')
                time_b = time.time()
                time_op = time_b - time_f
                total_time += time_op
                print(f'{idx}/{num_Var} clear {round(time_op,1)}s/{round(total_time,1)}')
                idx += 1

AEData_train.org()
AEData_test.org()

#%%
'''

FCL OAE

'''

m = FCCAutoencoder

Pre = [False,True]
norm = [None,'batchnorm','layernorm']
Nor = ['None','BN','LN']
num_layer = [n for n in range(1,6)]
num_Z = [1824,912,456,228,114,57,28,14,7,4,2]

num_Var = len(Pre)*len(norm)*len(num_layer)*len(num_Z)

criterion_Cos = nn.CosineEmbeddingLoss()
Ortho_loss_log = {}

idx = 1
total_time = 0 

for pre in Pre:
    if pre:
       AEData_train.std()
       AEData_test.std()
    for N,n in tqdm(enumerate(norm)):
        for l in num_layer:
            for z in num_Z:
                time_f = time.time()
                torch.cuda.empty_cache()
                plt.clf()
                AEData_train.cuda()
                title = f'FCL_OAE_Pre{pre}_{Nor[N]}_FCC_layer_num{l}_Z{z}'
                
                try:
                    model = train(FCCAutoencoder, AEData_train, z,100, title = title, num_layer = l,normalization = n,batch_size=2048,orthogonal=True)
                except:
                    model = train(FCCAutoencoder, AEData_train, z,100, title = title, num_layer = l,normalization = n,batch_size=256,orthogonal=True)
                AEData_train.cpu()
                model.eval()
                
                AEData_test.cuda()
                Ortho_loss_log[pre,Nor[N],l,z] = criterion_Cos(model(AEData_test()),AEData_test(),torch.tensor([1],device=device)).item()
                fig, axs = plt.subplots(2)
                fig.suptitle(title)
                axs[0].plot(AEData_test()[0].to('cpu'))
                axs[0].grid()
                Decode = model(AEData_test()[0:1]).detach().to('cpu')
                axs[1].plot(Decode.T)
                axs[1].grid()
                plt.show()
                np.savetxt(f'Result/FCL_OAE/Decoding/{title}.csv',Decode,delimiter=',')
                del Decode
                fig.savefig(f'Fig/FCL_OAE/{title}_Decoding',dpi=300)
                fig.clf()
                

                fig, axs = plt.subplots(2)
                fig.suptitle(title)
                axs[0].plot(Y_test)
                axs[0].grid()
                Encode = model.encode(AEData_test()).detach().to('cpu')
                axs[1].plot(Encode)
                axs[1].grid()
                plt.show()
                AEData_test().cpu()
                np.savetxt(f'Result/FCL_OAE/Encoding/{title}.csv',Encode,delimiter=',')
                del Encode
                fig.savefig(f'Fig/FCL_OAE/{title}_Encoding',dpi=300)
                fig.clf()
                
                torch.save(model,f'Model/FCL_OAE/{title}.pt')
                time_b = time.time()
                time_op = time_b - time_f
                total_time += time_op
                print(f'{idx}/{num_Var} clear {round(time_op,1)}s/{round(total_time,1)}')
                idx += 1

AEData_train.org()
AEData_test.org()
