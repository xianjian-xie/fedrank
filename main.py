import torch
import pickle
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cnn import CNN
from torch import nn

import os


device = "cuda" if torch.cuda.is_available() else "cpu"

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

learning_rate = 1e-3
batch_size = 128
n_batches = 100
n_epochs = 3
Nslaves = 5

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = CNN(1, 4*4*50).to(device)
# model = CNN().to(device)
print(model)

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
        # print(f"Epoch {t + 1}\n-------------------------------")
        prev_model = clonecoefs(model)
        train_epoch(dataloader, model, loss_fn, optimizer)
        print(f'Trained Epoch {t}')
        update = [(x-y).cpu()/learning_rate  for x,y in zip(clonecoefs(model), prev_model)]

        

        savecoeflist(update, f'history_{round}_{slaves}_{t}.pyc')
    testloss, correct =  test_loop(test_dataloader, model, loss_fn)
    return testloss, correct



def comm_round(dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, round):
    # start with the avged model in model.state_dict()
    clientcoeflist = []
    testloss_list = []
    correct_list = []

    for slaves in range(Nslaves):
        setcoefs(model, averagedcoefs)  # set averagedcoefs to model
        testloss, correct = client_action(dataloader, test_dataloader, model, loss_fn, optimizer, round, slaves)
        clientcoeflist.append(clonecoefs(model))
        testloss_list.append(testloss)
        correct_list.append(correct)

    testloss_tensor = torch.tensor(testloss_list)
    correct_tensor = torch.tensor(correct_list)

    print(f"Test Error: \n Accuracy: {(100*torch.mean(correct_tensor)):>0.1f}% ({(100*torch.std(correct_tensor)):>0.1f}%), Loss: {(100*torch.mean(testloss_tensor)):>8f} ({(100*torch.std(testloss_tensor)):>8f})\n")
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

for jj in range(500):
    averagedcoefs = comm_round(train_dataloader, test_dataloader, model, loss_fn, optimizer, averagedcoefs, jj)

