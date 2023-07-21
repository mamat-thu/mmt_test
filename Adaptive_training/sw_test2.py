import tensorflow as tf
#import tensorflow.examples.tutorials.mnist.input_data as input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt
import h5py




# data
num_class = 10

test_dict = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0501/test_hw.mat", 'r')
data = np.array(test_dict["data"]).astype(np.float32).transpose([2,1,0])
data = np.reshape(data[0:400,:,:],[-1,1024])

data_label = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0430/test_y.mat", 'r')
data_label = np.array(data_label["test_y"]).astype(np.int32).reshape(-1)

data_label = np.eye(num_class)[np.squeeze(data_label[0:400]).reshape(-1)].astype(np.float32)
data = np.concatenate((data, data_label), axis = 1)

print(data.shape, data_label.shape)

print("Load data done.")



total_img = np.shape(data)[0]
print ('total_img:',total_img)
#plt.imshow(data[1000,:1024].reshape([32,32]))
#plt.show()

weight_fc1_bin = np.load('./model_bin_fc1_108_3625%.npy')

fc1 = np.matmul(data[:,:1024],weight_fc1_bin)
fc1[fc1>=0] = 1
fc1[fc1<0] = -1
fc1 = fc1[:,0:10]

# fc2 = np.matmul(fc1, weight_fc2)
maxindex_pred = np.argmax(fc1,axis=1)
maxindex_label = np.argmax(data[:,1024:],axis=1)
pred_correct = np.sum(maxindex_pred==maxindex_label)
acc = pred_correct/total_img

print('correct:',pred_correct)
print ('prechange_acc:',acc)




