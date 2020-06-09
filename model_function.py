# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:14:05 2019

@author:shshen
"""

import tensorflow as tf
import numpy as np




def cal_att(inputs):
    

    xnorm = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=2))

    xnorm1 = tf.tile(tf.expand_dims(xnorm, 1), [1, tf.shape(inputs)[1], 1])
    xnorm2 = tf.tile(tf.expand_dims(xnorm, -1), [1, 1, tf.shape(inputs)[1]])

    x_x =  tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))

    outputs = tf.div(x_x, xnorm1 * xnorm2)
    diag_vals = tf.ones_like(outputs[0, :, :]) # (l, l)
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (l, l)
    sel = tf.eye(tf.shape(outputs)[1])
    sel2 = tf.eye(1, num_columns=tf.shape(outputs)[1])
    sel3 = tf.zeros([tf.shape(outputs)[1]-1, tf.shape(outputs)[1]])
    sel4 = tf.concat([sel2,sel3], axis = 0)
    sel = sel - sel4
    tril = tril - sel
    
    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, l, l)
   # masks=tf.linalg.band_part(masks,5,0)
    paddings = tf.ones_like(masks)*(-2**32+1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)


    outputs = tf.nn.softmax(outputs)

    return outputs




def GLU(inputs, dim, scope='GLU'):
    r = tf.nn.sigmoid(tf.layers.dense(inputs, dim))

    output = tf.layers.dense(inputs, dim)

    output = output * r
    return output


def cnn_block(x, filter1, k1, k2, drop_rate, is_training):

    o1 = x
    
    w1 = tf.Variable(tf.contrib.layers.xavier_initializer()([k1, filter1, filter1]),dtype=tf.float32)
    w2 = tf.zeros([k2, filter1, filter1], dtype=tf.float32)
    res_w = tf.concat([w1,w2],axis = 0)
    b = tf.Variable(tf.contrib.layers.xavier_initializer()([filter1]),dtype=tf.float32)
    o2 = tf.nn.conv1d(o1, res_w, 1, padding='SAME') + b
    o2 = tf.nn.dropout(o2, drop_rate)
    o2 = GLU(o2, filter1)
    return o2 + x
def CNN(x, skills, individual, next_skill, out_size, drop_rate, is_training):
    print(np.shape(x))
    att = cal_att(skills)
    x2 = tf.matmul(att, x)
    x = tf.concat([x,x2,individual],axis = -1)
    x = GLU(x, out_size)
    for i in range(1,4):
        x = cnn_block(x,out_size, 6, 6, drop_rate, is_training)
    x = x*next_skill
    return x
  
