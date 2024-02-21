import torch
# torch.manual_seed(2)
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
# import cv2
import copy
from data import AddGaussianNoise
from data import visualize_dataset, visualize_image,shuffle_dataset_target, add_noise_to_dataset, flip_noise_dataset, add_noise_to_model, check
import sys
from sklearn.decomposition import PCA
from numpy.linalg import norm
from numpy.linalg import inv

from sklearn.cluster import KMeans
import scipy.stats as ss
from permute_CUSUM_fix import Monitor

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

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
        kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
        label = kmeans.labels_
        
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

def find_minority_outlier_cluster(relation_list):
    X = relation_list.reshape((-1,1))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    label = kmeans.labels_
    cluster1_idx = np.where(label==0)[0].tolist()
    cluster2_idx = np.where(label==1)[0].tolist()
    if len(cluster1_idx) <= len(cluster2_idx):
        return cluster1_idx
    else:
        return cluster2_idx

# python main.py data monitor 3.84 0.4

attack_mode = sys.argv[1]
detect_mode = sys.argv[2]
H = sys.argv[3]
k = sys.argv[4]

root = os.getcwd()    
log_path =  os.path.join(root, 'plot_log', attack_mode + detect_mode + H + k)

H = float(H)
k = float(k)



device = "cuda" if torch.cuda.is_available() else "cpu"


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

mean= 0.01
std = 0.1

divisor = 250

data_poisoned_training_data = copy.deepcopy(training_data)
data_poisoned_training_data.transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean, std)
])
target_poisoned_training_data = shuffle_dataset_target(training_data, divisor)




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

    root = os.getcwd()    
    # file_path =  os.path.join(root, 'noattack230603', filename)
    # with open(file_path, 'wb') as f:
    #     pickle.dump(flattened, f)
    #     f.close()

def clonecoefs(model):
    lst = []
    for param in model.parameters():
        with torch.no_grad():
            lst.append(param.clone())
    return lst


def train_epoch(dataloader, model, loss_fn, optimizer):
    num = 0
    
    for batch, (X, y) in enumerate(dataloader):
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
            break
        num = num + 1



def client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves):
    for t in range(n_epochs):
        prev_model = clonecoefs(model)
        train_epoch(dataloader, model, loss_fn, optimizer)
        update = [(x-y).cpu()/learning_rate  for x,y in zip(clonecoefs(model), prev_model)]

        

        savecoeflist(update, f'history_{round}_{slaves}_{t}.pyc')
    testloss, correct =  test_loop(test_dataloader, model, loss_fn)
    return testloss, correct



def client_norm(common_received_model, clientcoeflist, testloss_list, correct_list, round, cumon):

    with open('average_250_arr.pyc', 'rb') as f:
        average_array = pickle.load(f)
        f.close()
    average_array = average_array.reshape((-1,1))


    np_array = np.zeros((average_array.shape[0],Nslaves))

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
        
    relation = []
    for i in range(np_array.shape[1]):

        norm_value = -norm(np_array[:,[i]])
        
        relation.append(norm_value)


    rank = ss.rankdata(relation, method='ordinal')
    for i in range(len(rank)):
        rank[i] = int(rank[i]) - 1

    
    output = cumon.newobs(rank)
    if output[0]==1 and output[1]>3:
        runlength_list.append(output[1])
        with open(log_path + '.pyc', 'wb') as f:
            pickle.dump(runlength_list, f)
            f.close()
        cumon.reset()

def client_cosine(common_received_model, clientcoeflist, testloss_list, correct_list, round, cumon):

    with open('average_250_arr.pyc', 'rb') as f:
        average_array = pickle.load(f)
        f.close()
    average_array = average_array.reshape((-1,1))


    np_array = np.zeros((average_array.shape[0],Nslaves))

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
    
        
    relation = []
    cosine_matrix = np.zeros((np_array.shape[1],np_array.shape[1]))
    for i in range(np_array.shape[1]):
        for j in range(np_array.shape[1]):
            cosine_matrix[i,j] = norm(np_array[:,[i]] - np_array[:,[j]])
    diagonal_matrix = np.diag(cosine_matrix)
    matrix_without_diagonal = cosine_matrix - diagonal_matrix


    for i in range(np_array.shape[1]):
        
        cosine_value = -np.mean(matrix_without_diagonal[[i],:], axis=1)[0]
        
        relation.append(np.round(cosine_value, decimals=2))



    rank = ss.rankdata(relation, method='ordinal')
    for i in range(len(rank)):
        rank[i] = int(rank[i]) - 1

    
    output = cumon.newobs(rank)
    if output[0]==1 and output[1]>3:
        runlength_list.append(output[1])
        cumon.reset()


def client_krum(common_received_model, clientcoeflist, testloss_list, correct_list, round, cumon):

    with open('average_250_arr.pyc', 'rb') as f:
        average_array = pickle.load(f)
        f.close()
    average_array = average_array.reshape((-1,1))


    np_array = np.zeros((average_array.shape[0],Nslaves))

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
    
   
    relation = []


    distance_matrix = np.zeros((np_array.shape[1],np_array.shape[1]))
    for i in range(np_array.shape[1]):
        for j in range(np_array.shape[1]):
            if i==j:
                distance_matrix[i,j] = np.power(10,10)
            else:
                distance_matrix[i,j] = norm(np_array[:,[i]] - np_array[:,[j]])


    for i in range(np_array.shape[1]):
        
        list_row_i = distance_matrix[i,:].tolist()
        list2 = list(set(list_row_i))
        list2.sort()
        sum_min3_value = sum(list2[0:3])
        krum_value = -sum_min3_value       
        relation.append(np.round(krum_value, decimals=2))



    rank = ss.rankdata(relation, method='ordinal')
    for i in range(len(rank)):
        rank[i] = int(rank[i]) - 1

    
    output = cumon.newobs(rank)
    if output[0]==1 and output[1]>3:
        runlength_list.append(output[1])
        with open(log_path + '.pyc', 'wb') as f:
            pickle.dump(runlength_list, f)
            f.close()
        cumon.reset()

        

def client_monitor(common_received_model,clientcoeflist, testloss_list, correct_list, round, cumon):
    col = 1
    with open('pca_model_50_1col_arr.pickle', 'rb') as f:
        pca = pickle.load(f)
        f.close()
    A = np.transpose(pca.components_)
    A = A.reshape((-1,col))
    np_array = np.zeros((A.shape[0],Nslaves))

    flattened_common = torch.tensor([j for x in common_received_model for j in list(torch.flatten(x))])
    flattened_common_arr = flattened_common.cpu().detach().numpy()
    flattened_common_arr = flattened_common_arr.reshape((-1,1))
    


    for i in range(len(clientcoeflist)):
        flattened = torch.tensor([j for x in clientcoeflist[i] for j in list(torch.flatten(x))])
        flattened_arr = flattened.cpu().detach().numpy()
        flattened_arr = flattened_arr.reshape((-1,1))
        np_array[:,[i]] = flattened_arr - flattened_common_arr
        
    relation = []
    for i in range(np_array.shape[1]):
        projection_vector = A.dot(inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(np_array[:,[i]]))
        residual = -norm(np_array[:,[i]] - projection_vector)

        relation.append(np.round(residual, decimals=2))


    rank = ss.rankdata(relation, method='ordinal')
    for i in range(len(rank)):
        rank[i] = int(rank[i]) - 1

    
    output = cumon.newobs(rank)
    if output[0]==0 and output[1]>3:
        runlength_list.append(output[1])
        with open(log_path + '.pyc', 'wb') as f:
            pickle.dump(runlength_list, f)
            f.close()
        cumon.reset()



    


def comm_round(attack_mode, dataloader, data_poisoned_train_dataloader, target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, round, cumon):
    # start with the avged model in model.state_dict()
    clientcoeflist = []
    testloss_list = []
    correct_list = []

    common_received_model = copy.deepcopy(model)
    setcoefs(common_received_model, averagedcoefs)
    if detect_mode == 'monitor':
        if round >= phase1:
            for slaves in range(Nslaves):
                setcoefs(model, averagedcoefs)  # set averagedcoefs to model
                if  slaves == 0:
                    if attack_mode == 'target':
                        print('enter target attack mode')
                        testloss, correct = client_action(target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                        clientcoeflist.append(clonecoefs(model))
                    elif attack_mode == 'data':
                        print('enter data attack mode')
                        testloss, correct = client_action(data_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                        clientcoeflist.append(clonecoefs(model))
                    elif attack_mode == 'model':
                        print('enter model attack mode')
                        mean = 0
                        std = 0.0001
                        testloss, correct = client_action(train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                        clientcoeflist.append(add_noise_to_model(clonecoefs(model), mean, std))
                    elif attack_mode == 'none':
                        print('enter none attack mode')
                        testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                        clientcoeflist.append(clonecoefs(model))
                    else:
                        print('monitor attacker not enter client_action function')
                elif slaves != 0:
                    testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                    clientcoeflist.append(clonecoefs(model))
                else:
                    print('monitor not enter client_action function')
                testloss_list.append(testloss)
                correct_list.append(correct)
        else:
            for slaves in range(Nslaves):
                setcoefs(model, averagedcoefs)  # set averagedcoefs to model
                testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                clientcoeflist.append(clonecoefs(model))
                testloss_list.append(testloss)
                correct_list.append(correct)
    else:

        for slaves in range(Nslaves):
            setcoefs(model, averagedcoefs)  # set averagedcoefs to model
            if  slaves == 1:
                if attack_mode == 'target':
                    print('enter target attack mode')
                    testloss, correct = client_action(target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                    clientcoeflist.append(clonecoefs(model))
                elif attack_mode == 'data':
                    print('enter data attack mode')
                    testloss, correct = client_action(data_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                    clientcoeflist.append(clonecoefs(model))
                elif attack_mode == 'model':
                    print('enter model attack mode')
                    mean = 0
                    std = 0.0001
                    testloss, correct = client_action(train_dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                    clientcoeflist.append(add_noise_to_model(clonecoefs(model), mean, std))
                elif attack_mode == 'none':
                    print('enter none attack mode')
                    testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                    clientcoeflist.append(clonecoefs(model))
                else:
                    print('attacker not enter client_action function')
            elif slaves != 1:
                testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
                clientcoeflist.append(clonecoefs(model))
            else:
                print('not enter client_action function')
            testloss_list.append(testloss)
            correct_list.append(correct)

    if detect_mode != 'none':


        if detect_mode == 'monitor' and round >= phase1:
            client_monitor(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)

        if detect_mode == 'norm':
            client_norm(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)

        if detect_mode == 'cosine':
            client_cosine(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)

        if detect_mode == 'krum':
            client_krum(clonecoefs(common_received_model), clientcoeflist, testloss_list, correct_list, round, cumon)
       


    testloss_tensor = torch.tensor(testloss_list)
    correct_tensor = torch.tensor(correct_list)

    
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



averagedcoefs = clonecoefs(model)



cumon = Monitor(5,H,k)
cumon.reset()

phase1 = 50
rounds = 100
if detect_mode == 'monitor':
    rounds += phase1


for jj in range(rounds):
    averagedcoefs = comm_round(attack_mode, train_dataloader, data_poisoned_train_dataloader, target_poisoned_train_dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, jj, cumon)
    if len(runlength_list) >=6:
        break
