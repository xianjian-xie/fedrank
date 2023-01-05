import pickle
with open('monitor.pyc', 'rb') as f:
    VL = pickle.load(f)

import matplotlib.pyplot as plt
import math
import numpy as np

print('VL lenth, shape is', len(VL), type(VL))
# VL is a list, lenth, shape is 339 torch.Size([15])
percent = [.001, .002, .005, .01, .02, .05]
p = np.zeros([len(VL),len(percent)]);
for ii in range(len(VL)):
    for jj in range(len(percent)):
        p[ii,jj] = sum (VL[ii].numpy().cumsum() / VL[ii].numpy().sum() < 1. - percent[jj])

for jj in range(len(percent)):
    plt.plot([len(VL[ii]) for ii in range(len(VL))], p[:,jj])


plt.show()