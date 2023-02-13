import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def read_accuracy(file_name, key_word, format):
    file = open(file_name,'r')
    list1 = []
    
    # search the line including accuracy
    for line in file:
        m1=re.search(key_word, line)
        if m1:
            n1=re.findall(format, line) # 正则表达式
            if n1 is not None:
                list1.append(n1) # 提取精度数字
                # print('list1 is',list1)
        # break
        
    file.close()
    arr1 = np.array(list1).astype(float)
    # print(arr1[0,:])
    return arr1


def plot_accuracy():
    x = np.arange(1,51)
    y1 = np.around(read_accuracy('plot_log/targetnodetect1.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:50,1], decimals=3)

    y2 = np.around(read_accuracy('plot_log/target.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:50,1], decimals=3)
    # y3 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-cos-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)
    # y4 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-mask-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)
    # y5 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_true_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)
    # y6 = np.around(read_accuracy('cs6.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,2], decimals=3)

    plt.plot(x,y1,'-',label='No Client Selection')
    plt.plot(x,y2,'-',label='PCA Method')
    # plt.plot(x,y3,'-',label='Cosine Similarity Method')
    # plt.plot(x,y4,'-',label='Binary Vector Method')
    # plt.plot(x,y5,'-',label='Pick Out All Adversarial Clients')
    # plt.plot(x,y6,'-',label='Binary Vector Method')
    
    plt.legend(loc='lower right')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Round')
    plt.show()
    
def read_time_to_accuracy(file_name, key_word, format):
    file = open(file_name,'r')
    list1 = []
    
    # search the line including accuracy
    for line in file:
        m1=re.search(key_word, line)
        if m1:
            n1=re.findall(format, line) # 正则表达式
            if n1 is not None:
                list1.append(n1) # 提取精度数字
        
    file.close()
    arr1 = np.array(list1).astype(float)
    return arr1


def read_time(file_name, key_word, format):
    file = open(file_name,'r')
    list1 = []
    
    # search the line including accuracy
    for line in file:
        m1=re.search(key_word, line)
        if m1:
            n1=re.findall(format, line) # 正则表达式
            if n1 is not None:
                list1.append(n1) # 提取精度数字
        
    file.close()
    arr1 = np.array(list1).astype(float)
    for i in range(arr1.shape[0]):
        arr1[i,0] = np.sum(arr1[0:i+1,0],axis=0)
    return arr1
    

def plot_time_to_accuracy():
    # x2 = np.around(read_time('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_channel-0.2_valid-acc-cluster_diff.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:10,0], decimals=3)
    # y2 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_channel-0.2_valid-acc-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:10,3], decimals=3)
    # x3 = np.around(read_accuracy('cs3.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    # y3 = np.around(read_accuracy('cs3.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,2], decimals=3)
    # x4 = np.around(read_accuracy('cs4.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    x4 = np.around(read_time('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-acc-cluster_diff.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    print('x4 is',x4)
    y4 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-acc-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)

    # x5 = np.around(read_accuracy('cs5-400epoch.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    x5 = np.around(read_time('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-acc-cluster_diff.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    y5 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-cos-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)

    # x6 = np.around(read_accuracy('cs6-400epoch.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)
    # x6 = np.around(read_time('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-acc-cluster_diff.txt', 'time cost', '[-+]?[0-9]+\.+[0-9]+')[0:101,0], decimals=3)

    # y6 = np.around(read_accuracy('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-mask-cluster_diff.txt', 'Test Epoch', '[-+]?[0-9]+\.+[0-9]+')[0:101,3], decimals=3)

    # plt.plot(x2,y2,'-',label='acc')
    plt.plot(x4,y4,'-',label='Validation Set Method')
    plt.plot(x5,y5,'-',label='Cosine Similarity Method')
    plt.plot(x6,y6,'-',label='Binary Vector Method')
    # plt.plot(x3,y3,'-',label='Pick Out All Adversarial Clients')
    # plt.plot(y2,x2,'-',label='No Client Selection')
    # plt.plot(y4,x4,'-',label='Validation Set Method')
    # plt.plot(y5,x5,'-',label='Cosine Similarity Method')
    # plt.plot(y6,x6,'-',label='Binary Vector Method')
    # plt.plot(y3,x3,'-',label='Pick Out All Adversarial Clients')
    plt.legend(loc='lower right')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Time')
    plt.show()



def read_roc(file_name,key_word1,key_word2,format1,format2):
    file = open(file_name,'r')
    list1 = []
    list_all = []
    arr_all = np.array(list_all).reshape((0,1))
    print(arr_all.shape)
    # list3 = []
    real_list = []
    real_arr = np.array(real_list).reshape((1,0))
    
    
    # search the line including accuracy
    loop_ten = 0
    for line in file:
        m1=re.search(key_word1, line)
        m2=re.search(key_word2, line)
        if m1:
            n1=re.findall(format1, line) # 正则表达式
            if n1 is not None:
                if loop_ten %10 != 9:
                    list1.append(n1) # 提取精度数字
                    loop_ten += 1
                else:
                    
                    list1.append(n1)
                    arr1 = np.array(list1).astype(float)
                    # print('arr1 shape',arr1.shape)
                    if key_word1 == 'output1 is':
                        arr1 = arr1[:,3].reshape((-1,1))
                    # print(arr1, arr1.shape) #(10,1)
                    max_val = np.amax(arr1, axis=0)
                    arr1 = arr1/max_val
                    # print(arr1)
                    arr_all = np.concatenate((arr_all, arr1), axis=0)
                    # print('arr_all is',arr_all)
                    list1 = []
                    loop_ten += 1
                    
                    
        elif m2:
            n2=re.findall(format2, line) # 正则表达式
            if n2 is not None:
                n2 = np.array(n2).astype(int).reshape((1,-1))
                n2 = n2[0,1:11].reshape((1,-1))
                real_arr = np.concatenate((real_arr,n2) , axis=1)  
    file.close()
    
    arr_all = np.transpose(arr_all)
    real_arr = real_arr + 1
    # print(arr_all, arr_all.shape)
    # print(real_arr, real_arr.shape)
    y_label = real_arr.reshape(-1)  # 非二进制需要pos_label
    y_pre = arr_all.reshape(-1)
    print(y_label.shape,y_pre.shape)
    return y_label, y_pre


def plot_detection_rate():

    
    y_label4, y_pre4 = read_roc('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-acc-cluster_diff.txt','logger mean','real list is','[-+]?[0-9]+\.+[0-9]+','[-+]?[0-9]+')
    fpr4, tpr4, thersholds4 = roc_curve(y_label4, y_pre4, pos_label=1)
    roc_auc4 = auc(fpr4, tpr4)
    # for i, value in enumerate(thersholds):
        # print("%f %f %f" % (fpr[i], tpr[i], value))
    y_label5, y_pre5 = read_roc('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-cos-cluster_diff.txt','output is','real list is','[-+]?[0-9]+\.+[0-9]+','[-+]?[0-9]+')
    fpr5, tpr5, thersholds5 = roc_curve(y_label5, y_pre5, pos_label=1)
    roc_auc5 = auc(fpr5, tpr5)

    y_label6, y_pre6 = read_roc('experiment_result/0_CIFAR10_wresnet28x2_100_0.1_iid_target_local-0.4_valid-mask-cluster_diff.txt','output1 is','real list is','[-+]?[0-9]+','[-+]?[0-9]+')
    fpr6, tpr6, thersholds6 = roc_curve(y_label6, y_pre6, pos_label=2)
    roc_auc6 = auc(fpr6, tpr6)
    
    
    
    plt.plot(fpr4, tpr4, '-.', label='Validation Set Method (area = {0:.2f})'.format(roc_auc4), lw=2)
    plt.plot(fpr5, tpr5, '--', label='Cosine Similarity Method (area = {0:.2f})'.format(roc_auc5), lw=2)
    plt.plot(fpr6, tpr6, ':', label='Binary Vector Method (area = {0:.2f})'.format(roc_auc6), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


    # plt.plot(x,y2,'-',label='No Client Selection')
    # plt.plot(x,y4,'-',label='Validation Set Method')
    # plt.plot(x,y5,'-',label='Cosine Similarity Method')
    # plt.plot(x,y6,'-',label='Binary Vector Method')
    # plt.plot(x,y3,'-',label='Pick Out All Adversarial Clients')
    # plt.legend(loc='lower right')






def main():
    plot_accuracy()
    # plot_time_to_accuracy()
    # plot_detection_rate()
    

if __name__ == '__main__':
    main()
