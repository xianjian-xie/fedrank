import pickle
with open('varlist_1000x5.pyc', 'rb') as f:
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
        # print('h1', VL[ii].numpy().cumsum() / VL[ii].numpy().sum() < 1. - percent[jj])
        p[ii,jj] = sum (VL[ii].numpy().cumsum() / VL[ii].numpy().sum() < 1. - percent[jj])

    # break

for jj in range(len(percent)):
    plt.plot([len(VL[ii]) for ii in range(len(VL))], p[:,jj],label="{0:.3f}".format(1-percent[jj]))

plt.xlabel('T x K')
plt.ylabel('# of Components')  
plt.title('# of Components vs T x K')
plt.legend(loc="upper left")
plt.savefig('figure4.png', dpi=500, bbox_inches='tight', pad_inches=0)

plt.show()

# print(VL[338].shape)