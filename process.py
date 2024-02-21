import pickle
import numpy as np
import scipy.stats as ss
import os
import matplotlib.pyplot as plt
import cv2
import copy
from data import AddGaussianNoise
from data import visualize_dataset, visualize_image,shuffle_dataset_target, add_noise_to_dataset, flip_noise_dataset, add_noise_to_model, load
import sys
from sklearn.decomposition import PCA
from numpy.linalg import norm
from numpy.linalg import inv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.stats as ss
# from permute_CUSUM_fix import Monitor
from numpy import linalg as la


attack_modes = ['target', 'data', 'model']
detect_modes = ['monitor', 'norm', 'cosine', 'krum']
Hks = [[3.84, 0.4],[3.28, 0.5],[2.77, 0.6]]

root = os.getcwd()   
result_path = os.path.join(root, 'result')

for attack_mode in attack_modes:
    for detect_mode in detect_modes:
        for Hk in Hks:
            print(attack_mode,detect_mode,Hk)
            H = Hk[0]
            k = Hk[1]
            print('h2')
            file_path =  os.path.join(root, 'plot_log', attack_mode + detect_mode + f'{H}' + f'{k}')
            ex = load(file_path + '.pyc')
            # ex += np.random.normal(loc=0,scale=0.1,size=len(ex))
            print('h4')
            with open(result_path+'.txt', 'a') as f:
                print(f'{attack_mode} {detect_mode} {H} {k}, mean: {np.mean(ex)}, std: {np.std(ex)}', file=f)
                # print(f'{attack_mode} {detect_mode} {H} {k}, {ex}', file=f)

                f.flush()

                    
