# -*- coding: utf-8 -*-
"""
Created on Fri June  30 06:25:01 2017

@author: Shaurya Rawat
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import sys
from sklearn.utils import shuffle
from sklearn.externals import joblib

path_train='training.csv'
path_test='test.csv'
lookup='IdLookupTable.csv'

# Exploratory Data Analysis
data=pd.read_csv('training.csv')
data.shape # 7049 images with 31 features
# Data will need to be reshaped to input it as a tensor. We will define a function for it.
data['Image']
data.columns
# The features are: (['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
#       'right_eye_center_y', 'left_eye_inner_corner_x',
#       'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
#       'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
#       'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
#       'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
#       'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
#       'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
#       'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
#       'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
#       'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
#       'mouth_right_corner_y', 'mouth_center_top_lip_x',
#       'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
#       'mouth_center_bottom_lip_y', 'Image'],
#      dtype='object')
data.count() # Not all the features have all the values. There are many missing values in most of the columns


# Learning Parameters
batch_size=64
eval_batch_size=64
image_size=96
num_channels=1
seed=None
num_labels=30
num_epochs=1000
validation_size=100
early_stop=100

# Functions to load data and evaluate in batches
def load_data(test=False):
    fname=path_test if test else path_train
    df=pd.read_csv(fname)
    cols=df.columns[:-1]
    df['Image']=df['Image'].apply(lambda im:np.fromstring(im,sep=' ')/255.0)
    df=df.dropna()
    X=np.vstack(df['Image'])
    X=X.reshape(-1,image_size,image_size,1)
    if not test:
        y=df[cols].values/96.0
        X,y=shuffle(X,y)
        joblib.dump(cols,'cols.pkl',compress=3)

    else:
        y=None
    return X,y
def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    size = data.shape[0]
    if size < eval_batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_labels), dtype=np.float32)
    for begin in range(0, size, eval_batch_size):
        end = begin + eval_batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[-eval_batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions
def error_measure(predictions,labels):
       return np.sum(np.power(predictions-labels,2))/(2*predictions.shape[0])

# training and test dataset
train_dataset,train_labels=load_data()
test_dataset,_=load_data(test=True)

# Validation set to evaluate accuracy
validation_dataset=train_dataset[:validation_size,...]
validation_labels=train_labels[:validation_size]
train_dataset=train_dataset[validation_size:,...]
train_labels=train_labels[validation_size:]

train_size=train_labels.shape[0]
train_size # 1940

# Graph Inputs
x=tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
y=tf.placeholder(tf.float32,shape=(batch_size,num_labels))
eval_node=tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))

weights_conv1=tf.Variable(tf.truncated_normal([5,5,num_channels,32],stddev=0.1,seed=seed))
biases_conv1=tf.Variable(tf.zeros([32]))
weights_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1,seed=seed))
biases_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
weights_fc1=tf.Variable(tf.truncated_normal([image_size//4*image_size//4*64,512],stddev=0.01,seed=seed))
biases_fc1=tf.Variable(tf.constant(0.1,shape=[512]))
weights_fc2=tf.Variable(tf.truncated_normal([512,512],stddev=0.01,seed=seed))
biases_fc2=tf.Variable(tf.constant(0.1,shape=[512]))
weights_fc3=tf.Variable(tf.truncated_normal([512,num_labels],stddev=0.01,seed=seed))
biases_fc3=tf.Variable(tf.constant(0.1,shape=[num_labels]))

# Convolutional Neural Network with 2 Convolutional layers and 3 fully connected layers
def conv_net(data,train=False):
       conv=tf.nn.conv2d(data,weights_conv1,strides=[1,1,1,1],padding='SAME')
       relu=tf.nn.relu(tf.nn.bias_add(conv,biases_conv1))
       pool=tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
       conv=tf.nn.conv2d(pool,weights_conv2,strides=[1,1,1,1],padding='SAME')
       relu=tf.nn.relu(tf.nn.bias_add(conv,biases_conv2))
       pool=tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
       pool_shape=pool.get_shape().as_list()
       reshape=tf.reshape(pool,[pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3]])
       hidden=tf.nn.relu(tf.matmul(reshape,weights_fc1)+biases_fc1)
       if train:
              hidden=tf.nn.dropout(hidden,0.5,seed=seed)
       hidden=tf.nn.relu(tf.matmul(hidden,weights_fc2)+biases_fc2)
       if train:
              hidden=tf.nn.dropout(hidden,0.5,seed=seed)
       return tf.matmul(hidden,weights_fc3)+biases_fc3

train_pred=conv_net(x,True)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(train_pred-y),1))
regularizers=(tf.nn.l2_loss(weights_fc1)+tf.nn.l2_loss(biases_fc1)+
              tf.nn.l2_loss(weights_fc2)+tf.nn.l2_loss(biases_fc2)+
              tf.nn.l2_loss(weights_fc3)+tf.nn.l2_loss(biases_fc3))
loss+=1e-7*regularizers
eval_prediction=conv_net(eval_node)
global_step=tf.Variable(0,trainable=False)
learning_rate=tf.train.exponential_decay(1e-3,global_step*batch_size,train_size,0.95,staircase=True)

optimizer=tf.train.AdamOptimizer(learning_rate,0.95).minimize(loss,global_step=global_step)

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)
loss_train_record=list()
loss_valid_record=list()
early_stopping=np.inf
early_stopping_epoch=0

current_epoch=0

while current_epoch < num_epochs:
        shuffled_index = np.arange(train_size)
        np.random.shuffle(shuffled_index)
        train_dataset = train_dataset[shuffled_index]
        train_labels = train_labels[shuffled_index]

        for step in range(int(train_size / batch_size)):
            offset = step * batch_size
            batch_data = train_dataset[offset:(offset + batch_size), ...]
            batch_labels = train_labels[offset:(offset + batch_size)]
            feed_dict = {x: batch_data,
                         y: batch_labels}
            _, loss_train, current_learning_rate = sess.run([optimizer, loss, learning_rate], feed_dict=feed_dict)

        eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_node)
        loss_valid = error_measure(eval_result, validation_labels)

        print ('Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f' % (
            current_epoch,
            loss_train, loss_valid,
            loss_train / loss_valid,
            current_learning_rate
        ))
        loss_train_record.append(np.log10(loss_train))
        loss_valid_record.append(np.log10(loss_valid))
        sys.stdout.flush()

        if loss_valid < early_stopping:
            best_valid = loss_valid
            best_valid_epoch = current_epoch
        elif best_valid_epoch + early_stop < current_epoch:
            print("Early stopping.")
            print("Best loss was {:.6f} @ epoch {}.".format(early_stopping, early_stopping_epoch))
            break

        current_epoch += 1

print('Training Finished')










































