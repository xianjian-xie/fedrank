import torch
torch.manual_seed(0)
import pickle
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cnn import CNN
from torch import nn

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from data import AddGaussianNoise
from data import visualize_dataset, visualize_image,shuffle_dataset_target, add_noise_to_dataset, flip_noise_dataset, add_noise_to_model
import sys
from sklearn.decomposition import PCA
from numpy.linalg import norm
from numpy.linalg import inv
from sklearn.cluster import KMeans
import scipy.stats as ss
from permute_CUSUM_fix import Monitor




def transimg(img):
    # img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    npimg1 = np.transpose(npimg,(1,2,0)) # C*H*W => H*W*C
    return npimg1

def find_extremum(relation_list, tag):
    outlier_idx = []
    if tag == 'low':
        idx = np.argmin(relation_list)
    elif tag == 'high':
        idx = np.argmax(relation_list)
    outlier_idx.append(idx)
    return outlier_idx

def find_outlier1(relation_list, tag):
        X = np.array(relation_list).reshape((-1,1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        label = kmeans.labels_
        # print('label', label)
        # print('center', kmeans.cluster_centers_, kmeans.cluster_centers_.shape)
        
        if tag == 'high':
            if kmeans.cluster_centers_[0,0] < kmeans.cluster_centers_[1,0]:
                outlier_idx = np.where(label==1)[0]
            else:
                outlier_idx = np.where(label==0)[0]
        elif tag == 'low':
            if kmeans.cluster_centers_[0,0] < kmeans.cluster_centers_[1,0]:
                outlier_idx = np.where(label==0)[0]
            else:
                outlier_idx = np.where(label==1)[0]
        outlier_idx = outlier_idx.tolist()
        return outlier_idx

# python main.py data monitor 3.84 0.4

print('num arg is', len(sys.argv))
print('arg is', sys.argv)
attack_mode = sys.argv[1]
detect_mode = sys.argv[2]
H = sys.argv[3]
k = sys.argv[4]

root = os.getcwd()    
print('root is', root)
log_path =  os.path.join(root, 'plot_log', attack_mode + detect_mode + H + k)

H = float(H)
k = float(k)
print('attack mode is', attack_mode, type(attack_mode))
print('detect_mode is', detect_mode, type(detect_mode))
print('H is', H, type(H))
print('k is', k, type(k))
# detect, nodetect



device = "cuda" if torch.cuda.is_available() else "cpu"

print('device is', device)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    # transform=None
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

std = 100
mean= 100
ratio=1

data_poisoned_training_data = copy.deepcopy(training_data)
data_poisoned_training_data.transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean, std)
])
target_poisoned_training_data = shuffle_dataset_target(training_data)

# visualize_dataset(training_data)
# visualize_dataset(data_poisoned_training_data)
# visualize_dataset(target_poisoned_training_data)



# print('h1',training_data.targets)
# print('h1',training_data[0][1])

# dataset_poisoned_training_data = add_noise_to_dataset(training_data, mean, std, ratio)
# dataset_poisoned_training_data.transform = transforms.Compose([
#                 transforms.ToTensor()
#             ])
# visualize_dataset(dataset_poisoned_training_data)

# print('h2',id(dataset_poisoned_training_data)) # use id function to get address of object




learning_rate = 1e-3
batch_size = 128
n_batches = 100
n_epochs = 3
Nslaves = 5

num_detect = 0
runlength_list = []





train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
data_poisoned_train_dataloader = DataLoader(data_poisoned_training_data, batch_size=batch_size, shuffle=True)
target_poisoned_train_dataloader = DataLoader(target_poisoned_training_data, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = CNN(1, 4*4*50).to(device)
# model = CNN().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)




# averaged is a list, where each item represents parameters of a layer
def average_model(coeflist):
    averaged = []
    Nslaves = len(coeflist)
    Nlayers = len(coeflist[0])
    for idx in range(Nlayers):
        layer_param = coeflist[0][idx].clone()
        for j in range(1,Nslaves):
            if torch.any(torch.isnan(coeflist[j][idx])):
                print (f'nan found in slave {j} layer {idx}\n')
            layer_param = layer_param + coeflist[j][idx]
            # print('layer_param shape is', coeflist[j][idx].shape)
        averaged.append(layer_param/Nslaves)
    return averaged

# set coefs to model
def setcoefs(model, coefs):
    idx = 0
    for param in model.parameters():
        d = coefs[idx]
        with torch.no_grad():
            param.copy_(d)  # copy d to param
        idx += 1

def savecoeflist (coeflist, filename):
    flattened = torch.tensor([j for x in coeflist for j in list(torch.flatten(x))])
    # print('flattened shape is', flattened.shape)
    # # flattened shape is torch.Size([431080])

    root = os.getcwd()    
    # print('root is', root)
    file_path =  os.path.join(root, 'new_history', filename)
    # print('file_path',file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(flattened, f)
        f.close()

    # with open(filename, 'wb') as f:
    #     pickle.dump(flattened, f)
    #     f.close()

def clonecoefs(model):
    lst = []
    for param in model.parameters():
        # print('clone param shape is',param.shape)
        with torch.no_grad():
            lst.append(param.clone())
    return lst


def train_epoch(dataloader, model, loss_fn, optimizer):
    num = 0
    # print('dataloader is', dataloader, type(dataloader))
    # dataloader is <torch.utils.data.dataloader.DataLoader object at 0x00000147DF5FDC10> <class 'torch.utils.data.dataloader.DataLoader'>  
    
    for batch, (X, y) in enumerate(dataloader):
        # print('x is', X.shape)
        # print('y is', y.shape)
        # x is torch.Size([128, 1, 28, 28])
        # y is torch.Size([128])
        # print('batch idx is', batch)
        # visualize_image(X[0],y[0])
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if num >= n_batches-1:
            # print('batch is', batch) # batch is 99
            break
        num = num + 1



def client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves):
    for t in range(n_epochs):
        # print(f"Epoch {t + 1}\n-------------------------------")
        prev_model = clonecoefs(model)
        train_epoch(dataloader, model, loss_fn, optimizer)
        print(f'Slave Idx {slaves}, Trained Epoch {t}')
        update = [(x-y).cpu()/learning_rate  for x,y in zip(clonecoefs(model), prev_model)]

        

        savecoeflist(update, f'history_{round}_{slaves}_{t}.pyc')
    testloss, correct =  test_loop(test_dataloader, model, loss_fn)
    return testloss, correct


def client_monitor(common_received_model,clientcoeflist, testloss_list, correct_list, round, cumon):
    with open('pca_model_50_1col_arr.pickle', 'rb') as f:
        pca = pickle.load(f)
        f.close()
    A = np.transpose(pca.components_)
    A = A.reshape((-1,1))
    np_array = np.zeros((A.shape[0],Nslaves))
    # print('two shape is',new_good_np_array.shape,np_array.shape)

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        # print('flattened shape is', flattened.shape)
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
    

    # with open('np_array.pyc', 'wb') as f:
    #     pickle.dump(np_array, f)
    #     f.close()

        
    relation = []
    for i in range(np_array.shape[1]):
        projection_vector = A.dot(inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(np_array[:,[i]]))
        residual = -norm(np_array[:,[i]] - projection_vector)

        # print('A shape', A.shape, np_array[:,i].shape)
        # with open('subspace50.pyc', 'wb') as f:
        #     pickle.dump(np_array[:,i].reshape((-1,1))- A, f)
        # residual = -norm(np_array[:,i].reshape((-1,1))- A)
        
        print('output is tensor({}), {}'.format(i,residual))
        relation.append(residual)

    print('relation', relation)
    with open(log_path  + '.txt', 'a') as f:
        print('residual list: {}'.format(relation), file=f)
        f.flush()
    rank = ss.rankdata(relation)
    for i in range(len(rank)):
        rank[i] = int(rank[i]) - 1
    print('rank is', rank)
    with open(log_path  + '.txt', 'a') as f:
        print('rank list: {}'.format(rank), file=f)
        f.flush()

    
    
    
    output = cumon.newobs(rank)
    cumon.status()
    print('output is', output)
    if output[0] is not None:
        print('attacker found, idx: {}, time: {}'.format(output[0], output[1]))
        with open(log_path  + '.txt', 'a') as f:
            print('attacker found, idx: {}, time: {}'.format(output[0], output[1]), file=f)
            f.flush()
        runlength_list.append(output[1])
        with open(log_path  + '.txt', 'a') as f:
            print('runlength list is: {}'.format(runlength_list), file=f)
            f.flush()
        with open(log_path + '.pyc', 'wb') as f:
            pickle.dump(runlength_list, f)
            f.close()
        cumon.reset()


    



def client_selection(common_received_model,clientcoeflist, testloss_list, correct_list, round):
    # print('enter client_selection, len(list) = {}'.format(len(clientcoeflist)))
    # # print('common received is', len(common_received_model), type(common_received_model),common_received_model[0],type(common_received_model[0]))
    # # print('client coef is', len(clientcoeflist), type(clientcoeflist),len(clientcoeflist[0]),type(clientcoeflist[0]))
    # print('correct is', len(correct_list), type(correct_list[0]))
    # print('testloss is', len(testloss_list),type(testloss_list[0]))
    # # enter client_selection, len(list) = 5
    # # correct is 5 <class 'float'>
    # # testloss is 5 <class 'float'>

    with open('subspacearr.pyc', 'rb') as f:
        new_good_np_array = pickle.load(f)
        f.close()
    new_good_np_array = new_good_np_array
    np_array = np.zeros((new_good_np_array.shape[0],Nslaves))
    # print('two shape is',new_good_np_array.shape,np_array.shape)

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        # print('flattened shape is', flattened.shape)
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
    
    relation = []
    for i in range(np_array.shape[1]):
        # new_np_array = pca.fit_transform(np_array)
        cosine = np.dot(np_array[:,i],new_good_np_array)/(norm(np_array[:,i])*norm(new_good_np_array))
        # print("cliend id, Cosine Similarity:", i, cosine)
        print('output is tensor({}), {}'.format(i,cosine))
        relation.append(cosine)

    print('relation', relation)

    outlier_idx = find_outlier1(relation, 'low')

    print('outlier_idx is', outlier_idx)


    global num_detect
    for i in range(len(outlier_idx)):
        clientcoeflist.pop(outlier_idx[i])
        testloss_list.pop(outlier_idx[i])
        correct_list.pop(outlier_idx[i])
        if outlier_idx[i] == 0 or outlier_idx[i] == 1:
            num_detect = num_detect + 1

    # print('lenth is', len(clientcoeflist))

    


def comm_round(attack_mode, dataloader, data_poisoned_train_dataloader, target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, round, cumon):
    # start with the avged model in model.state_dict()
    clientcoeflist = []
    testloss_list = []
    correct_list = []

    common_received_model = copy.deepcopy(model)
    setcoefs(common_received_model, averagedcoefs)
    for slaves in range(Nslaves):
        setcoefs(model, averagedcoefs)  # set averagedcoefs to model
        if 0 <= slaves <= 0:
            if attack_mode == 'target':
                print('enter target attack mode')
                testloss, correct = client_action(target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
            elif attack_mode == 'data':
                print('enter data attack mode')
                testloss, correct = client_action(data_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
            elif attack_mode == 'model':
                print('enter model attack mode')
                mean = 0
                std = 0.01
                testloss, correct = client_action(train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                clientcoeflist.append(add_noise_to_model(clonecoefs(model), mean, std))
            elif attack_mode == 'none':
                print('enter none attack mode')
                testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
            else:
                print('attacker not enter client_action function')
        elif 1 <= slaves <= 4 :
            testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
        else:
            print('not enter client_action function')
        if attack_mode != 'model' or slaves != 0:
            clientcoeflist.append(clonecoefs(model))
        print('round {}, slave {}, correct {}, loss {}'.format(round,slaves,correct,testloss))
        testloss_list.append(testloss)
        correct_list.append(correct)

    if detect_mode == 'detect':
        client_selection(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round)

    if detect_mode == 'monitor':
        client_monitor(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)

    # if detect_mode == 'norm':
    #     client_monitor(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)


    testloss_tensor = torch.tensor(testloss_list)
    correct_tensor = torch.tensor(correct_list)

    # print(f"Test Error: \n Accuracy: {(100*torch.mean(correct_tensor)):>0.1f}% ({(100*torch.std(correct_tensor)):>0.1f}%), Loss: {(torch.mean(testloss_tensor)):>8f} ({(torch.std(testloss_tensor)):>8f})\n")
    print('Test Epoch: {}(100%)  Loss: 1.8124  Accuracy: {}'.format(round, 100*torch.mean(correct_tensor) ))
    with open(log_path + '.txt', 'a') as f:
        print('Test Epoch: {}(100%)  Loss: 1.8124  Accuracy: {}'.format(round, 100*torch.mean(correct_tensor) ), file=f)
        f.flush()
    
    # central server average the model
    updatedcoefs = average_model(clientcoeflist)
    round += 1
    return updatedcoefs


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct

# for i in range(len(clonecoefs(model))):
#     print('model list[{}] shape is {}'.format(i, clonecoefs(model)[i].shape))



averagedcoefs = clonecoefs(model)



cumon = Monitor(5,H,k)
cumon.reset()

for jj in range(100):
    averagedcoefs = comm_round(attack_mode, train_dataloader, data_poisoned_train_dataloader, target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, jj, cumon)
    print('num detect', num_detect)
    # with open(log_path, 'a') as f:
    #     print('num detect', num_detect, file=f)
    #     f.flush()

