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
import scipy.stats as ss
# from permute_CUSUM_fix import Monitor
from numpy import linalg as la

# ex1 = [5, 6, 5, 6, 4, 3, 5, 3]
# ex2 = [4, 3, 4, 4, 4, 5, 5, 4, 5, 6, 5, 5, 3, 5, 5, 3, 4]
# ex3 = [5, 5, 5, 4, 4, 6, 5, 4, 4, 3, 5, 3, 5, 3, 4, 4, 4, 7, 2]

# print(np.mean(ex1),np.std(ex1), np.mean(ex2),np.std(ex2),np.mean(ex3),np.std(ex3))

ex1 = [6, 5, 6, 3, 2, 4, 3, 3, 5, 4, 5, 5]
ex2 = [4, 4, 3, 5, 4, 3, 3, 5, 5, 6]
ex3 = [6, 6, 5, 3, 2, 5, 3, 2, 5, 3, 6, 3]

print(np.mean(ex1),np.std(ex1), np.mean(ex2),np.std(ex2),np.mean(ex3),np.std(ex3))

ex1 = [4, 4, 5, 3, 4, 3, 5, 4, 4, 4, 5]
ex2 = [5, 4, 4, 4, 4, 5, 4, 5, 5, 6]
ex3 = [3, 5, 4, 4, 6, 4, 3, 7, 6, 4, 2, 4, 3]

print(np.mean(ex1),np.std(ex1), np.mean(ex2),np.std(ex2),np.mean(ex3),np.std(ex3))

# A = np.array([[3,5,5],[5,3,5],[-5,-5,-7]])
# good_np_array = np.array([[-1,-1,0,2,0],[-2,0,0,1,1]])
# # xx = 0.2* (x.dot(np.transpose(x)))
# # xx = 0.2 * (np.transpose(x).dot(x))
# # pca = PCA(n_components = 0.9)
# pca = PCA(n_components = 1)
# good_np_array = np.transpose(good_np_array)
# pca.fit(good_np_array)
# print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
# new_good_np_array = pca.fit_transform(good_np_array)
# print('new good np array shape',new_good_np_array.shape, new_good_np_array)
# print('components is', pca.components_)

# with open('pca.pickle', 'wb') as f:
#     pickle.dump(pca, f)

# with open('pca.pickle', 'rb') as f:
#     pca = pickle.load(f)
# print('num of components, explained variance are', pca.n_components_,pca.explained_variance_ratio_)
# print('components is', pca.components_, pca.components_.shape)


# w, v = la.eig(xx)
# print('w is',w)
# print('v is',v)




# with open('pca_model_50_1col_arr.pickle', 'rb') as f:
#     pca = pickle.load(f)
# with open('np_array.pyc', 'rb') as f:
#     np_array = pickle.load(f)

# A = np.transpose(pca.components_)

# print('shape is', A.shape, np_array[:,[0]].shape)

# relation = []
# for i in range(np_array.shape[1]):
#     projection_vector = A.dot(inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(np_array[:,[i]]))
#     # projection_vector = A.dot(np.transpose(A).dot(np_array[:,[i]]))
#     residual = norm(np_array[:,[i]] - projection_vector)
#     relation.append(residual)

# print('residual list is',relation)


# list1 = [2,5,4]
# rank = ss.rankdata(list1)
# print('rank is', rank)

# p=10
# r = np.random.permutation(p)
# z = np.random.rand(p)
# print('r is',r)
# print('z is',z)

# arr1 = np.zeros((10,5))
# arr2 = np.ones((10,1))

# print('shape',arr1.shape, arr2.shape)

# arr1[:,[0]] = arr2

# print(arr1,arr1.shape)



# with open('subspace.pyc', 'rb') as f:
#     VL = pickle.load(f)

# print('vl shape', VL.shape)

# nn=12

# def f1():
#     print('nn is',nn)
#     nn = nn+1
# f1()
# print('nn2',nn)

# def select(str1,n):
#     str1.pop(1)
#     n=n+1

# n=1
# str1=[1,2,3,4,5,6]
# print('shape is',str1.shape)
# # str1.pop(2)
# print('str1',str1,n)
# select(str1,n)
# print('str1',str1,n)



# str2=['abc','bcd','dce']
# del str2[1]
# print('str2',str2)
