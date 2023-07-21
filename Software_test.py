import tensorflow as tf
#import tensorflow.examples.tutorials.mnist.input_data as input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt

data = []
data.append(np.load('zero_label.npy'))
data.append(np.load('one_label.npy'))
data.append(np.load('two_label.npy'))
data.append(np.load('three_label.npy'))
data.append(np.load('four_label.npy'))
data.append(np.load('five_label.npy'))
data.append(np.load('six_label.npy'))
data.append(np.load('seven_label.npy'))
data.append(np.load('eight_label.npy'))
data.append(np.load('nine_label.npy'))

'''
data = np.concatenate((zero,one),axis=0)
data = np.concatenate((data,two),axis=0)
data = np.concatenate((data,three),axis=0)
data = np.concatenate((data,four),axis=0)
data = np.concatenate((data,five),axis=0)
data = np.concatenate((data,six),axis=0)
data = np.concatenate((data,seven),axis=0)
data = np.concatenate((data,eight),axis=0)
data = np.concatenate((data,nine),axis=0)


print (np.shape(data))
'''

#total_img = np.shape(data)[0]
#print ('total_img:',total_img)
#plt.imshow(data[1000,:1024].reshape([32,32]))
#plt.show()

total_img = np.zeros([10])
correct_img = np.zeros([10])

weight_fc1_bin = np.load('model_bin_fc1_98_87%.npy')
weight_fc2 = np.load('model_fc2_98_87%.npy')

for i in range(10):
  fc1 = np.matmul(data[i][:,:1024],weight_fc1_bin)
  fc1[fc1>=0] = 1
  fc1[fc1<0] = -1
  fc2 = np.matmul(fc1, weight_fc2)
  maxindex_pred = np.argmax(fc2,axis=1)
  maxindex_label = np.argmax(data[i][:,1024:],axis=1)
  correct_img[i] = np.sum(maxindex_pred==maxindex_label)
  total_img[i] = np.shape(data[i])[0]
  print (i,":",correct_img[i]/total_img[i])

print ('all_acc:',np.sum(correct_img)/np.sum(total_img))
'''
all_sum = np.zeros(16)
for i in range(16):
  all_sum[i] = np.sum(weight_fc1_bin[:,i])
  print (all_sum[i])
  if (all_sum[i] > 88):
    weight_fc1_bin[(1024-(abs(int(all_sum[i]/4.2)))):,i]=-1
    weight_fc1_bin[0:(abs(int(all_sum[i]/4.2))),i]=-1
  if (all_sum[i] < -88):
    weight_fc1_bin[(1024-(abs(int(all_sum[i]/4.2)))):,i]=1
    weight_fc1_bin[0:(abs(int(all_sum[i]/4.2))),i]=1

#weight_temp = np.reshape(weight_fc1_bin[:,11],[32,32])
#weight_temp[25:,:] = 1
#weight_fc1_bin[:,11] = np.reshape(weight_temp,[1024])

fc1 = np.matmul(data[:,:1024],weight_fc1_bin)
fc1[fc1>=0] = 1
fc1[fc1<0] = -1
fc2 = np.matmul(fc1, weight_fc2)
maxindex_pred = np.argmax(fc2,axis=1)
maxindex_label = np.argmax(data[:,1024:],axis=1)
pred_correct = np.sum(maxindex_pred==maxindex_label)
acc = pred_correct/total_img

print ('afterchange_acc:',acc)

for i in range(16):
  all_sum[i] = np.sum(weight_fc1_bin[:,i])
  print (all_sum[i])

np.save('model_bin_fc1_'+str(int(np.max(abs(all_sum))))+'_'+str(int(acc*100))+'%.npy', weight_fc1_bin)
np.save('model_fc2_'+str(int(np.max(abs(all_sum))))+'_'+str(int(acc*100))+'%.npy', weight_fc2)
'''
