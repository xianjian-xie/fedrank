import torch
import pickle
import copy
from sklearn.decomposition import PCA
import numpy as np



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

def f2():
    print('enter f2')
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
        if ii==49:
            with open('subspace50.pyc', 'wb') as f:
                pickle.dump(x, f)

        # XX = torch.matmul(x, x.transpose(1,0))
        # L, V = torch.linalg.eig(XX)
        # L_real = L.real
        # # print('L real is', L_real)
        # VarList.append(L_real)


    # with open('monitor.pyc', 'wb') as f:
    #     pickle.dump(VarList, f)

def f3():
    print('enter f3')
    with open('subspace50.pyc', 'rb') as f:
        VL = pickle.load(f)

    print('vl shape', VL.shape)
    good_np_array = VL.cpu().detach().numpy()
    # good_np_array = np.transpose(good_np_array)
    print('good_np_array shape', good_np_array.shape)

    # pca = PCA(n_components = 0.90)
    pca = PCA(n_components = 1)
    pca.fit(good_np_array)
    print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
    # new_good_np_array = pca.fit_transform(good_np_array)
    new_good_np_array = pca.fit_transform(good_np_array)
    print('new good np array shape',new_good_np_array.shape)
    print('components shape',pca.components_.shape, type(pca.components_))

    with open('subspace50_1col_arr.pyc', 'wb') as f:
        pickle.dump(new_good_np_array, f)
    with open('pca_model_50_1col_arr.pickle', 'wb') as f:
        pickle.dump(pca, f)

    

if __name__ == "__main__":
    f3()