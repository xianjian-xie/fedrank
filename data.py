import torch
torch.manual_seed(0)
import pickle
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cnn import CNN
from torch import nn

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import random


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def visualize_dataset(dataset):
    # image_array,y=dataset[i]
    image_array,y=dataset[0]
    # image_array=noisy_dataset.data[i]
    # print('visualize is', image_array)
    print(image_array.size())
    image_array=image_array.reshape(28,28)
    # print(image_array)
    plt.imshow(image_array)
    plt.title('label {}'.format(y))
    plt.show()

def visualize_image(image_array,y):
    # image_array,y=dataset[i]
    # image_array,y=dataset[0]
    # image_array=noisy_dataset.data[i]
    # print('visualize is', image_array)
    print(image_array.size())
    image_array=image_array.reshape(28,28)
    # print(image_array)
    plt.imshow(image_array)
    plt.title('label {}'.format(y))
    plt.show()



def shuffle_dataset_target(dataset):
    shuffled_dataset = copy.deepcopy(dataset)
    # for i in range(len(shuffled_dataset.target)):
    #     print('shuffle is', shuffled_dataset.data[i].shape, type(shuffled_dataset.data[i]))
    #     shuffle is (32, 32, 3) <class 'numpy.ndarray'>
    for i in range(len(shuffled_dataset.targets)):
        # print('target is', shuffled_dataset.target[i], type(shuffled_dataset.target[i]))
        # shuffled_dataset.target[i] = shuffled_dataset.target[(i+1)%len(shuffled_dataset.target)]
        # shuffled_dataset.target[i] = (shuffled_dataset.target[i]+1)%10
        shuffled_dataset.targets[i] = (shuffled_dataset.targets[i]+random.randint(0,9))%10
        # print('shuffled target is', shuffled_dataset.target[i], type(shuffled_dataset.target[i]))

    return shuffled_dataset

def add_noise_to_dataset(dataset, mean, std, ratio):
    print('ratio is', ratio)
    noisy_dataset = copy.deepcopy(dataset)
    print('noisy_dataset start is', noisy_dataset[0][0])
    print('length is', int(len(noisy_dataset.targets)*ratio))
    print('length1 is', noisy_dataset.train_data[0,:,:])
    noisy_dataset.data = noisy_dataset.data.type(torch.float64)
    # print('original dataset shape is', noisy_dataset.data[0].shape, noisy_dataset.data[0])
    # for i in range(dataset.data.shape[0]):
    for i in range(1):    
        print('size is',noisy_dataset.data[i].size())
        plt.imshow(noisy_dataset.data[i])
        plt.show()
        # noise = np.random.normal(mean, std, noisy_dataset.data[i].shape)
        noise = torch.randn(noisy_dataset.data[i].size()) * std + mean
        print('noise is',noise)
        # noise = noise.astype(np.uint8)
        # print('baocuo2')
        # if (i == 0):
            # print('noise shape is', noise.shape, noise)
        # noisy_dataset.data[i] = noisy_dataset.data[i].type(torch.float32)
        noisy_dataset.data[i] = noisy_dataset.data[i]/255.0 + noise
        noisy_dataset.data[i] = torch.clamp(noisy_dataset.data[i], min=0, max=1)
        noisy_dataset.data[i] = noisy_dataset.data[i] * 255 
        noisy_dataset.data[i] = noisy_dataset.data[i].type(torch.uint8)
        print('data after noise',noisy_dataset.data[i])
        # noisy_dataset.data[i] = noisy_dataset.data[i] + noise
        # noisy_dataset.data[i] = noisy_dataset.data[i] + torch.randn(noisy_dataset.data[i].size()) * std + mean
        plt.imshow(noisy_dataset.data[i])
        plt.show()
        # noisy_dataset[i][0] = noisy_dataset[i][0] + noise
        # noisy_dataset.data[i] = noisy_dataset.data[i].astype(np.uint8)

        # # image_array,y=dataset[i]
        # image_array,y=noisy_dataset[i]
        # # image_array=noisy_dataset.data[i]
        # print('image array is', image_array)
        # print(image_array.size())
        # image_array=image_array.reshape(28,28)
        # # print(image_array)
        # plt.imshow(image_array)
        # plt.show()
        # image_array = image_array + torch.randn(image_array.size()) * std + mean
        # print('image_array_data is', image_array)
        # plt.imshow(image_array)
        # plt.show()
        # image_array = image_array * 255
        # print('image_array_data2 is', image_array)
        # plt.imshow(image_array)
        # plt.show()
    
    # # image_array,y=dataset[i]
    # image_array,y=noisy_dataset[0][0]
    # # image_array=noisy_dataset.data[i]
    # print('image array is', image_array)
    # print(image_array.size())
    # image_array=image_array.reshape(28,28)
    # # print(image_array)
    # plt.imshow(image_array)
    # plt.show()


    # print('noisy dataset shape is', noisy_dataset.data[0].shape, noisy_dataset.data[0])
    return noisy_dataset

def add_noise_to_model(model, mean, std):
    # print('ratio is', ratio)
    # print('model is',type(model),model)
    noisy_model = copy.deepcopy(model)
    for i in range(len(noisy_model)):
        noise = torch.randn(noisy_model[i].size()) * std + mean
        noisy_model[i] = noisy_model[i] + noise

    # for k, v in noisy_model.items():
    #     noise = np.random.normal(mean, std, noisy_model[k].shape)
    #     noisy_model[k] = (noisy_model[k]/torch.max(noisy_model[k]) + noise) * torch.max(noisy_model[k])
    #     # noisy_model.state_dict()[k] = noisy_model.state_dict()[k] + torch.randn(noisy_model.state_dict()[k].size()) * std + mean
    return noisy_model
    #      return tensor + torch.randn(tensor.size()) * self.std + self.mean
    #     noisy_model.state_dict()[k] = 
    #     model_m_dict[k]
    #     noisy_modelmodel_m.state_dict().items()
    # for i in range(len(noisy_model.target)):
    #     noise = np.random.normal(mean, std, noisy_dataset.data[i].shape)
    #     noisy_dataset.data[i] = noisy_dataset.data[i] + noise
    # return noisy_dataset

def flip_noise_dataset(dataset, mean, std):
    flipped_noisy_dataset = copy.deepcopy(dataset)
    for i in range(len(flipped_noisy_dataset.target)//2):
        # print('target is', shuffled_dataset.target[i], type(shuffled_dataset.target[i]))
        # shuffled_dataset.target[i] = shuffled_dataset.target[(i+1)%len(shuffled_dataset.target)]
        flipped_noisy_dataset.target[i] = (flipped_noisy_dataset.target[i]+1)%10
        # print('shuffled target is', shuffled_dataset.target[i], type(shuffled_dataset.target[i]))
    for i in range(len(flipped_noisy_dataset.target)//2, len(flipped_noisy_dataset.target)):

        noise = np.random.normal(mean, std, flipped_noisy_dataset.data[i].shape)
        flipped_noisy_dataset.data[i] = (flipped_noisy_dataset.data[i]/255 + noise)*255
        flipped_noisy_dataset.data[i] = flipped_noisy_dataset.data[i].astype(np.uint8)

    return flipped_noisy_dataset




def make_batchnorm_dataset(dataset):
    dataset = copy.deepcopy(dataset)
    plain_transform = datasets.Compose([transforms.ToTensor(),
                                        transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = plain_transform
    return dataset


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


