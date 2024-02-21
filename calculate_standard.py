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


# backup block
def standard():
    print('enter standard')
    ii = 0
    jj = 0
    kk = 0

    with open(f'./monitor/history_0_0_0.pyc', 'rb') as f:
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
    for ii in range(50):
        for jj in range(5):
            aggr_update =  torch.zeros(history_standard.size())
            for kk in range(3):
                with open(f'./monitor/history_{ii}_{jj}_{kk}.pyc','rb') as f:
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


    # with open('monitor.pyc', 'wb') as f:
    #     pickle.dump(VarList, f)

# record first 50's round gradient update
def f2():
    print('enter f2')
    ii = 0
    jj = 0
    kk = 0

    with open(f'./monitor5/history_0_0_0.pyc', 'rb') as f:
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
    for ii in range(100):
        for jj in range(5):
            aggr_update =  torch.zeros(history_standard.size())
            for kk in range(3):
                with open(f'./monitor5/history_{ii}_{jj}_{kk}.pyc','rb') as f:
                    newrow = pickle.load(f)
                    f.close()
                aggr_update = aggr_update + newrow.view(1,-1)
            # print('aggr_update size is', aggr_update.size())
            # # aggr_update size is torch.Size([1, 431080])
            x = torch.cat((x, aggr_update.view(1,-1)),0)
        if ii==99:
            with open('subspace100_monitor5.pyc', 'wb') as f:
                pickle.dump(x, f)
                # output subspace50.pyc, subspace50-1.pyc, subspace50-2.pyc

        # XX = torch.matmul(x, x.transpose(1,0))
        # L, V = torch.linalg.eig(XX)
        # L_real = L.real
        # # print('L real is', L_real)
        # VarList.append(L_real)


    # with open('monitor.pyc', 'wb') as f:
    #     pickle.dump(VarList, f)

# use first 50's round gradient update to calculate pca
def f3():
    print('enter f3')
    with open('subspace50.pyc', 'rb') as f:
        VL = pickle.load(f)

    print('vl shape', VL.shape) 
    # vl shape is torch.Size([250, 431080])
    good_np_array = VL.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array.shape)

    # pca = PCA(n_components = 0.95)
    pca = PCA(n_components = 2)
    pca.fit(good_np_array)
    print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
    # new_good_np_array = pca.fit_transform(good_np_array)
    new_good_np_array = pca.fit_transform(good_np_array)
    print('new good np array shape',new_good_np_array.shape)
    print('components shape',pca.components_.shape, type(pca.components_))

    # vl shape torch.Size([250, 431080])
    # good_np_array shape (250, 431080)
    # num of components, explained variance are 1 [0.84486526]
    # new good np array shape (250, 1)
    # components shape (1, 431080) <class 'numpy.ndarray'>

    with open('subspace50_2col_arr.pyc', 'wb') as f:
        pickle.dump(new_good_np_array, f)
    with open('pca_model_50_2col_arr.pickle', 'wb') as f:
        pickle.dump(pca, f)

# preparation for norm comparison method
def f4():
    print('enter f4')
    with open('subspace50.pyc', 'rb') as f:
        VL = pickle.load(f)

    print('vl shape', VL.shape)
    # vl shape is torch.Size([250, 431080])
    good_np_array = VL.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array.shape)

    sum_array = np.zeros((1, good_np_array.shape[1]))
    for i in range(good_np_array.shape[0]):
        sum_array = sum_array + good_np_array[i,:]
    average_array = sum_array/250

    with open('average_250_arr.pyc', 'wb') as f:
        pickle.dump(average_array, f)

# preparion for pca+ cluster comparison method
def f5():
    print('enter f5')
    with open('subspace100_monitor5.pyc', 'rb') as f:
        VL = pickle.load(f)

    print('vl shape', VL.shape) 
    # vl shape is torch.Size([250, 431080])
    good_np_array = VL.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array.shape)

    # pca = PCA(n_components = 0.95)
    pca = PCA(n_components = 2)
    pca.fit(good_np_array)
    print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
    # new_good_np_array = pca.fit_transform(good_np_array)
    new_good_np_array = pca.fit_transform(good_np_array)
    print('new good np array shape',new_good_np_array.shape)
    print('components shape',pca.components_.shape, type(pca.components_))

    with open('subspace100_monitor5_2col_arr.pyc', 'wb') as f:
        pickle.dump(new_good_np_array, f)
    with open('pca_model_100_monitor5_2col_arr.pickle', 'wb') as f:
        pickle.dump(pca, f)

    # output: subspace50_2col_arr.pyc, subspace50_2col_arr-1.pyc, subspace50_2col_arr-2.pyc
    # pca_model_50_2col_arr.pickle, pca_model_50_2col_arr-1.pickle, pca_model_50_2col_arr-2.pickle

def f6():
    print('enter f6')
    with open('subspace50_2col_arr.pyc', 'rb') as f:
        VL = pickle.load(f)
    # vl shape is torch.Size([250, 431080])
    good_np_array = VL
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array.shape)
    # print(good_np_array)
    with open('subspace50_2col_arr-1.pyc', 'rb') as f:
        VL1 = pickle.load(f)
    good_np_array1 = VL1
    print('good_np_array1 shape', good_np_array1.shape)

    with open('subspace50_2col_arr-2.pyc', 'rb') as f:
        VL2 = pickle.load(f)
    good_np_array2 = VL2
    print('good_np_array2 shape', good_np_array2.shape)

    with open('subspace50_2col_arr-3.pyc', 'rb') as f:
        VL3 = pickle.load(f)
    good_np_array3 = VL3
    print('good_np_array3 shape', good_np_array3.shape)

    var_arr = np.zeros((50,2))
    for i in range(50):
        var1 = np.var(good_np_array[i*5:(i+1)*5,0],axis=0)
        var1_1 = np.var([good_np_array[i*5,0],good_np_array1[i*5,0],good_np_array2[i*5,0],good_np_array3[i*5,0]])
        mean1 = np.mean(good_np_array[i*5:(i+1)*5,0],axis=0)
        mean1_1 = np.mean([good_np_array[i*5,0],good_np_array1[i*5,0],good_np_array2[i*5,0],good_np_array3[i*5,0]])
        var2 = np.var(good_np_array[i*5:(i+1)*5,1],axis=0)
        var2_2 = np.var([good_np_array[i*5,1],good_np_array1[i*5,1],good_np_array2[i*5,1],good_np_array3[i*5,1]])
        mean2 = np.mean(good_np_array[i*5:(i+1)*5,1],axis=0)
        mean2_2 = np.mean([good_np_array[i*5,1],good_np_array1[i*5,1],good_np_array2[i*5,1],good_np_array3[i*5,1]])
        print('mean is', mean1, mean1_1, mean2, mean2_2)
        print('var is',  var1, var1_1, var2, var2_2)
        var_arr[i,0] = var1
        var_arr[i,1] = var2
    # var1 = np.var(good_np_array[:,0],axis=0)
    # var2 = np.var(good_np_array[:,1],axis=0)
    # print('var is', var1, var2)

    # print('difference', good_np_array-good_np_array2)
    plt.hist(good_np_array[:,0],bins=30)
    plt.show()


# save gradient update of monitor6-like experiment(round0-49 normal, round50-99 attack)
#  round0-round49, and round0-round99 
def f7():
    print('enter f7')
    ii = 0
    jj = 0
    kk = 0

    with open(f'./monitor6/history_0_0_0.pyc', 'rb') as f:
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
    for ii in range(100):
        for jj in range(5):
            aggr_update =  torch.zeros(history_standard.size())
            for kk in range(3):
                with open(f'./monitor6/history_{ii}_{jj}_{kk}.pyc','rb') as f:
                    newrow = pickle.load(f)
                    f.close()
                aggr_update = aggr_update + newrow.view(1,-1)
            # print('aggr_update size is', aggr_update.size())
            # # aggr_update size is torch.Size([1, 431080])
            x = torch.cat((x, aggr_update.view(1,-1)),0)
        if ii==49:
            with open('subspace50_monitor6.pyc', 'wb') as f:
                pickle.dump(x, f)
        if ii==99:
            with open('subspace100_monitor6.pyc', 'wb') as f:
                pickle.dump(x, f)
                # output subspace50.pyc, subspace50-1.pyc, subspace50-2.pyc
                # output subspace50_monitor6.pyc, subspace100_monitor6.pyc

        # XX = torch.matmul(x, x.transpose(1,0))
        # L, V = torch.linalg.eig(XX)
        # L_real = L.real
        # # print('L real is', L_real)
        # VarList.append(L_real)


    # with open('monitor.pyc', 'wb') as f:
    #     pickle.dump(VarList, f)

def f8():
    print('enter f8')
    with open('subspace50_monitor6.pyc', 'rb') as f:
        VL50 = pickle.load(f)

    with open('subspace100_monitor6.pyc', 'rb') as f:
        VL100 = pickle.load(f)


    print('vl shape', VL50.shape) 
    # vl shape is torch.Size([250, 431080])
    good_np_array50 = VL50.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array50.shape)

    print('vl shape', VL100.shape) 
    # vl shape is torch.Size([250, 431080])
    good_np_array100 = VL100.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array100.shape)


    # pca = PCA(n_components = 0.95)
    pca = PCA(n_components = 2)
    # new_good_np_array = pca.fit_transform(good_np_array)
    new_good_np_array = pca.fit_transform(good_np_array50)
    print('new good np array shape',new_good_np_array.shape)
    print('components shape',pca.components_.shape, type(pca.components_))

    pca.fit(good_np_array50)
    print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
    new_good_np_array1 = pca.transform(good_np_array100)
    print('new good np array shape',new_good_np_array1.shape)
    print('components shape',pca.components_.shape, type(pca.components_))


    with open('subspace50_monitor6_2col_arr.pyc', 'wb') as f:
        pickle.dump(new_good_np_array, f)
    with open('pca_model_50_monitor6_2col_arr.pickle', 'wb') as f:
        pickle.dump(pca, f)

    with open('subspace100_monitor6_2col_arr.pyc', 'wb') as f:
        pickle.dump(new_good_np_array1, f)
    with open('pca_model_100_monitor6_2col_arr.pickle', 'wb') as f:
        pickle.dump(pca, f)

    # output: subspace50_2col_arr.pyc, subspace50_2col_arr-1.pyc, subspace50_2col_arr-2.pyc
    # pca_model_50_2col_arr.pickle, pca_model_50_2col_arr-1.pickle, pca_model_50_2col_arr-2.pickle




if __name__ == "__main__":
    f3()