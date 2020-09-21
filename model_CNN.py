# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

from model_function import *

class CKT(object):

    def __init__(self, batch_size, num_steps, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_steps = num_steps
        self.num_skills =  num_skills

        self.input_data = tf.placeholder(tf.int32, [None, num_steps], name="input_data")
        self.input_skill = tf.placeholder(tf.int32, [None, num_steps], name="input_skill")
        self.l = tf.placeholder(tf.float32, [None, num_steps, num_skills], name="l")
        self.next_id = tf.placeholder(tf.int32, [None, num_steps], name="next_id")
        self.target_id = tf.placeholder(tf.int32, [None], name="target_id")
        self.target_correctness = tf.placeholder(tf.float32, [None], name="target_correctness")
        self.target_id2 = tf.placeholder(tf.int32, [None], name="target_id2")
        self.target_correctness2 = tf.placeholder(tf.float32, [None], name="target_correctness2")
        
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.contrib.layers.xavier_initializer()

       # input_data = tf.cast(self.input_data, tf.float32)

        skill_w = tf.Variable(tf.contrib.layers.xavier_initializer()([num_skills,100]),dtype=tf.float32, trainable=True)
        skills = tf.nn.embedding_lookup(skill_w, self.input_skill)
        next_skill = tf.nn.embedding_lookup(skill_w, self.next_id)

        zeros = tf.zeros([num_skills,100])
        t1 = tf.concat([skill_w,zeros],axis = -1)
        t2 = tf.concat([zeros,skill_w],axis = -1)
        input_w = tf.concat([t1,t2],axis = 0)
        input_data = tf.nn.embedding_lookup(input_w, self.input_data)
        
        
        outputs = CNN(input_data, skills, self.l, next_skill, self.hidden_size, self.dropout_keep_prob, self.is_training)

        print(np.shape(outputs))
        
        logits = tf.reduce_sum(outputs, axis = -1)
        print('aa')
        print(np.shape(logits))

        self.states = tf.sigmoid(logits, name="states")
        # from output nodes to pick up the right one we want
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.target_id)
        selected_logits2 = tf.gather(logits, self.target_id2)

        #make prediction
        self.pred = tf.sigmoid(selected_logits, name="pred")

        # loss function
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=self.target_correctness), name="losses")
        
        self.cost = self.loss
