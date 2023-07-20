import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py



'''prepare data'''

num_class = 10

# data
train_dict = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0501/train_x.mat", 'r')
data = np.array(train_dict["data"]).astype(np.float32).transpose([2,1,0])
data = np.reshape(data[0:2400,:,:],[-1,1024])

data_label = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0430/train_y.mat", 'r')
data_label = np.array(data_label["train_y"]).astype(np.int32).transpose([1,0])
data_label = np.eye(num_class)[np.squeeze(data_label[0:2400,:]).reshape(-1)].astype(np.float32)

data = np.concatenate((data, data_label), axis = 1)

print(data.shape, data_label.shape)

test_dict = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0501/test_x.mat", 'r')
data_test = np.array(test_dict["data"]).astype(np.float32).transpose([2,1,0])
data_test = np.reshape(data_test[0:400,:,:],[-1,1024])

data_test_label = h5py.File("/home/mmt/workspace/DATA_SET/KMNIST_0430/test_y.mat", 'r')
data_test_label = np.array(data_test_label["test_y"]).astype(np.int32).reshape(-1)

data_test_label = np.eye(num_class)[np.squeeze(data_test_label[0:400]).reshape(-1)].astype(np.float32)
data_test = np.concatenate((data_test, data_test_label), axis = 1)

print(data_test.shape, data_test_label.shape)

print("Load data done.")





print ('data_train_shape:',np.shape(data))
print ('data_test_shape:',np.shape(data_test))

total_img = np.shape(data)[0]
print ('total_img:',total_img)
#plt.imshow(data[1000,:1024].reshape([32,32]))
#plt.show()

np.random.shuffle(data)
import pickle
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

def weight_variable(shape):
    initial =  tf.random.truncated_normal(shape, stddev=0.4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.compat.v1.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)



with tf.device('/gpu:1'):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.InteractiveSession(
              config=tf.compat.v1.ConfigProto(
              log_device_placement=False,
              allow_soft_placement=True,
              gpu_options=gpu_options,
              ))
    ckpt_dir = './BWN_tf_ckpt'
    restore = True
    Train = True

    if Train==True:
        restore = False


    
    x = tf.compat.v1.placeholder("float", shape=[None, 1024])
    y_ = tf.compat.v1.placeholder("float", shape=[None, 10])

    W_fc1 = weight_variable([1024, 16])
    W_fc1 = tf.clip_by_value(W_fc1,-256,256)
    bin_W_fc1 = binarize(W_fc1)
    #b_fc1 = bias_variable([16])
    #h_fc1 = tf.matmul(x, bin_W_fc1) + b_fc1
    h_fc1 = tf.matmul(x, bin_W_fc1)
    h_fc1 = tf.clip_by_value(h_fc1,-256,256)
    bin_h_fc1 = binarize(h_fc1)

    bin_h_fc1 = bin_h_fc1[:,0:10]
    y = tf.nn.softmax(bin_h_fc1)

#    W_fc2 = weight_variable([16, 10])
#    #W_fc2 = tf.clip_by_value(W_fc2,-1.5,1.5)
#    #bin_W_fc2 = binarize(W_fc2)
#    y = tf.nn.softmax(tf.matmul(bin_h_fc1,W_fc2))

    lr_decay_step=10000
    start_lr=0.01
    global_steps = tf.Variable(0,trainable=False)

    cross_entropy = -tf.reduce_sum(y_*tf.math.log(y))
    lr = tf.compat.v1.train.exponential_decay(start_lr, global_steps, lr_decay_step, 0.2, staircase=True)
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy,global_step=global_steps)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver = tf.compat.v1.train.Saver()

    max_abs_sum = 200
    max_abs_sum_pre = 400
    img_index = 0
    batch_num = 1024
    max_test_acc = 0
    max_train_acc = 0
    while(max_abs_sum>10):

        sess.run(tf.compat.v1.global_variables_initializer())
        #save_path = saver.save(sess,"./model.ckpt")
        if restore==True:
            ckpt_model = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, ckpt_model)
            print ('Model restore done!')


        if Train==True:
            for i in range(10000):
                if img_index+batch_num>=total_img:
                    np.random.shuffle(data)
                    img_index=0
                train_img = data[img_index:(img_index+batch_num),:1024]
                train_label = data[img_index:(img_index+batch_num),1024:]
                img_index+=batch_num

                train_step.run(feed_dict={x: train_img, y_: train_label})

                if i%200 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: data[:,:1024], y_: data[:,1024:]})
                    test_accuracy = accuracy.eval(feed_dict={
                        x:data_test[:,:1024], y_: data_test[:,1024:]})

                    if train_accuracy > max_train_acc:
                        max_train_acc = train_accuracy

                    if test_accuracy > max_test_acc:
                        max_test_acc = test_accuracy

                        weight=sess.run(bin_W_fc1)
                        all_sum = np.zeros(16)
                        for k in range(16):
                            all_sum[k] = np.sum(weight[:,k])
                            # print (all_sum[i])
                        max_abs_sum = np.max(abs(all_sum))
                        if max_test_acc > 0.2:
                            np.save('./output_2400/model_bin_fc1_'+str(int(max_test_acc*10000))+'%'+'_'+str(int(max_abs_sum))+'.npy',weight)
                            # np.save('./output/model_fc2_'+str(int(max_test_acc*10000))+'_'+str(int(max_abs_sum))+'%.npy',sess.run(W_fc2))


                    print ("step %d, training accuracy %g, test accuracy %g, learning rate %g, max_train_acc %g, max_test_acc %g" 
                            %(i, train_accuracy,test_accuracy,sess.run(lr), max_train_acc, max_test_acc))

            saver.save(sess, ckpt_dir+'/ckpt',global_step=i)
            print ('Model save done!')

        acc = accuracy.eval(feed_dict={x: data[:,:1024], y_: data[:,1024:]})
        print ("train accuracy %g"%(acc))

        acc_test = accuracy.eval(feed_dict={x: data_test[:,:1024], y_: data_test[:,1024:]})
        print ("test accuracy %g, max_test_acc %g"%(acc_test, max_test_acc))

        weight=sess.run(bin_W_fc1)

        all_sum = np.zeros(16)
        for i in range(16):
            all_sum[i] = np.sum(weight[:,i])
            print (all_sum[i])
        max_abs_sum = np.max(abs(all_sum))
        print ('max_abs_sum:',max_abs_sum)
        if max_abs_sum<max_abs_sum_pre:
            # np.save('./model_bin_fc1_'+str(int(max_abs_sum))+'_'+str(int(acc*10000))+'_maxtest'+str(int(max_test_acc*10000))+'%.npy',weight)
            # np.save('./model_fc2_'+str(int(max_abs_sum))+'_'+str(int(acc*10000))+'%.npy',sess.run(W_fc2))
            max_abs_sum_pre=max_abs_sum





