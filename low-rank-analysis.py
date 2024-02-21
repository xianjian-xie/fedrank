
import torch

import pickle
import numpy as np
import scipy.stats as ss
import os
import matplotlib.pyplot as plt
import cv2
import copy
from data import AddGaussianNoise
from data import visualize_dataset, visualize_image,shuffle_dataset_target, add_noise_to_dataset, flip_noise_dataset, add_noise_to_model
import sys
from sklearn.decomposition import PCA
from numpy.linalg import norm
from numpy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.stats as ss
# from permute_CUSUM_fix import Monitor
from numpy import linalg as la


# Professor used this function to generate low rank property graph
def standard():
    # with open(f'./histories/history_54_2_2.pyc', 'rb') as f:
    #     x = pickle.load(f)
    #     f.close()

    # print('x shape is',x.shape)

    ii = 0
    jj = 0
    kk = 0

    with open(f'./histories/history_0_0_0.pyc', 'rb') as f:
        x = pickle.load(f)
        f.close()
    # x shape is torch.Size([431080])

    d = x.size(dim=0)
    x = torch.empty((0,d))

    VarList = []

    print(f"Test file processed. Vector size {d}\n")
    for ii in range(500):
        for jj in range(5):
            for kk in range(3):
                with open(f'./histories/history_{ii}_{jj}_{kk}.pyc','rb') as f:
                    newrow = pickle.load(f)
                    f.close()
                x = torch.cat((x, newrow.view(1,-1)),0)
        XX = torch.matmul(x, x.transpose(1,0)) 
        VarList.append(XX.eig().eigenvalues[:,0])   
        # append real part of all eigenvalues, varlist is is list, 1st item 1 x m,
        # m equals num of eigenvalue for XX generated in 1st communication round
        


    # with open('varlist.pyc', 'wb') as f:
    #    pickle.dump(VarList, f)


def b1():
    print('enter standard')
    ii = 0
    jj = 0
    kk = 0

    with open(f'./noattack230603/history_0_0_0.pyc', 'rb') as f:
        x = pickle.load(f)
        f.close()
    # x shape is torch.Size([431080])
    history_standard = copy.deepcopy(x)
    history_standard = history_standard.view(1,-1)
    print('history_standard size is', history_standard.size())
    d = x.size(dim=0)
    x = torch.empty((0,d))

    VarList = []

    print(f"Test file processed. Vector size {d}\n")
    for ii in range(500):
        for jj in range(5):
            aggr_update =  torch.zeros(history_standard.size())
            for kk in range(3):
                with open(f'./noattack230603/history_{ii}_{jj}_{kk}.pyc','rb') as f:
                    newrow = pickle.load(f)
                    f.close()
                aggr_update = aggr_update + newrow.view(1,-1)
            # print('aggr_update size is', aggr_update.size())
            # # aggr_update size is torch.Size([1, 431080])
            x = torch.cat((x, aggr_update.view(1,-1)),0)
        XX = torch.matmul(x, x.transpose(1,0))
        L, V = torch.linalg.eig(XX)
        L_real = L.real
        # print('L real is', L_real)
        VarList.append(L_real)

    with open('varlist_xie_230603.pyc', 'wb') as f:
       pickle.dump(VarList, f)



if __name__ == "__main__":
    b1()
