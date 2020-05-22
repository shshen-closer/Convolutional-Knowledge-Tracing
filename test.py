# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import tensorflow as tf
from sklearn import metrics
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import checkmate as cm
from utils import data_helpers as dh
import csv
# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))

MODEL = input("Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")


TESTSET_DIR = 'data/assist2009_updated_all.csv'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_float("l2_lambda", 0.0001, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.003, "Learning rate")
tf.flags.DEFINE_float("norm_ratio", 10, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.flags.DEFINE_integer("batch_size", 1, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 50, "Number of epochs to train for.")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test():

    # Load data
    logger.info("Loading data...")

    logger.info("Training data processing...")

    test_students, test_max_num_problems, test_max_skill_num = dh.read_test_data_from_csv_file(FLAGS.test_data_file)
    max_num_steps = test_max_num_problems
    max_num_skills = test_max_skill_num
    

    # Load rnn model
    BEST_OR_LATEST = input("Load Best or Latest Model?(B/L): ")

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_data = graph.get_operation_by_name("input_data").outputs[0]
            input_skill = graph.get_operation_by_name("input_skill").outputs[0]
            l  = graph.get_operation_by_name("l").outputs[0]
            next_id = graph.get_operation_by_name("next_id").outputs[0]
            target_id = graph.get_operation_by_name("target_id").outputs[0]
            target_correctness = graph.get_operation_by_name("target_correctness").outputs[0]
            target_id2 = graph.get_operation_by_name("target_id2").outputs[0]
            target_correctness2 = graph.get_operation_by_name("target_correctness2").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            skill = graph.get_operation_by_name("skill_w").outputs[0]
            states = graph.get_operation_by_name("states").outputs[0]
            pred = graph.get_operation_by_name("pred").outputs[0]
            
            data_size = len(test_students)
            index = 0
            actual_labels = []
            pred_labels = []
            while(index+FLAGS.batch_size < data_size):
                x = np.zeros((FLAGS.batch_size, max_num_steps))
                xx = np.zeros((FLAGS.batch_size, max_num_steps))
                next_id_b = np.zeros((FLAGS.batch_size, max_num_steps))
                l_b = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills))
                target_id_b = []
                target_correctness_b = []
                target_id2_b = []
                target_correctness2_b = []
                
                for i in range(FLAGS.batch_size):
                    student = test_students[index+i]
                    problem_ids = student[1]
                    correctness = student[2]
                    leng = len(problem_ids)

                    correct_num = np.zeros(max_num_skills)
                    answer_count = np.ones(max_num_skills)
                    for j in range(len(problem_ids)-1):
                        problem_id = int(problem_ids[j])
                        
                        if(int(correctness[j]) == 0):
                            x[i, j] = problem_id + max_num_skills
                        else:
                            x[i, j] = problem_id
                            correct_num[problem_id] += 1
                        l_b[i,j] = correct_num / answer_count
                        answer_count[problem_id] += 1
                        xx[i,j] = problem_id
                        next_id_b[i,j] = int(problem_ids[j+1])

                        target_id_b.append(i*max_num_steps+j)
                        target_correctness_b.append(int(correctness[j+1]))
                        actual_labels.append(int(correctness[j+1]))
                    target_id2_b.append(i*max_num_steps+j)
                    target_correctness2_b.append(int(correctness[j+1]))

                index += FLAGS.batch_size

                feed_dict = {
                    input_data: x,
                    input_skill: xx,
                    l: l_b,
                    next_id: next_id_b,
                    target_id: target_id_b,
                    target_correctness: target_correctness_b,
                    target_id2: target_id2_b,
                    target_correctness2: target_correctness2_b,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                
                '''
                skill_b = sess.run([skill], feed_dict)
                print(np.shape(skill_b))
                item = skill_b[0]
                with open('skill_2009.txt', 'a')as fi:
                    for temp in item:
                        for iii in temp:
                            fi.write(str(iii) + ',')
                        fi.write('\n')
                break
                '''
                pred_b, state = sess.run([pred, states], feed_dict)
                print(np.shape(state))
                print(np.shape(pred_b))
                state = np.squeeze(state, axis = 0)
                state = state[:leng]
                
                if leng >50 and leng<100:
                    writer = csv.writer(open('state.csv', 'a', newline=''))
                    writer.writerow([len(problem_ids)])
                    writer.writerow(student[1])
                    writer.writerow(student[2])
                    writer.writerow(state)
                    writer.writerow('\n')
                for p in pred_b:
                    pred_labels.append(p)
            rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
            fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            #calculate r^2
            r2 = r2_score(actual_labels, pred_labels)
            print("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g} ".format(1,rmse, auc, r2))
            logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g} ".format(1,rmse, auc, r2))

    logger.info("Done.")


if __name__ == '__main__':
    test()
