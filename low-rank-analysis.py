import torch
import pickle

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

