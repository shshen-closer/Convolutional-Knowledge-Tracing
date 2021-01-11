# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import tensorflow as tf
from datetime import datetime 
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_CNN import CKT
from utils import checkmate as cm
from utils import data_helpers as dh

# Parameters
# ==================================================

TRAIN_OR_RESTORE = 'T' #input("Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()).replace(':', '_'))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

name = str(sys.argv[1])
number = str(sys.argv[2])
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.flags.DEFINE_float("learning_rate", 0.003, "Learning rate")
tf.flags.DEFINE_float("norm_ratio", 5, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_size", 100, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 50 , "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 50, "Number of epochs to train for.")

tf.flags.DEFINE_string("train_data_path", 'data/'+ name+ '_train' + number + '.csv', "Path to the training dataset")
tf.flags.DEFINE_string("valid_data_path", 'data/'+ name+ '_valid' + number + '.csv', "Path to the testing dataset")


tf.flags.DEFINE_integer("decay_steps",10, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.2, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def train():
    """Training model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_students, train_max_num_problems, train_max_skill_num = dh.read_data_from_csv_file(FLAGS.train_data_path)
    
    logger.info("Validation data processing...")
    test_students, test_max_num_problems, test_max_skill_num = dh.read_data_from_csv_file(FLAGS.test_data_path)
    max_num_steps = max(train_max_num_problems, test_max_num_problems)
    max_num_skills = max(train_max_skill_num, test_max_skill_num)
    
    # Build a graph and lstm_3 object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            ckt = CKT(
                batch_size = FLAGS.batch_size,
                num_steps = max_num_steps,
                num_skills = max_num_skills,
                hidden_size = FLAGS.hidden_size, 
                )
            

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=ckt.global_step, decay_steps=(len(train_students)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
                #grads, vars = zip(*optimizer.compute_gradients(ckt.loss))
                #grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                #train_op = optimizer.apply_gradients(zip(grads, vars), global_step=ckt.global_step, name="train_op")
                train_op = optimizer.minimize(ckt.loss, global_step=ckt.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", ckt.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load ckt model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())



            current_step = sess.run(ckt.global_step)

            def train_step(x, xx, l, next_id, target_id, target_correctness, target_id2, target_correctness2):
                """A single training step"""

                #print(ability)
                feed_dict = {
                    ckt.input_data: x,
                    ckt.input_skill: xx,
                    ckt.l: l,
                    ckt.next_id: next_id,
                    ckt.target_id: target_id,
                    ckt.target_correctness: target_correctness,
                    ckt.target_id2: target_id2,
                    ckt.target_correctness2: target_correctness2,
                    ckt.dropout_keep_prob: FLAGS.keep_prob,
                    ckt.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, ckt.global_step, train_summary_op, ckt.pred, ckt.loss], feed_dict)

                
                logger.info("step {0}: loss {1:g} ".format(step,loss))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step(x, xx, l, next_id, target_id, target_correctness, target_id2, target_correctness2):
                """Evaluates model on a validation set"""

                feed_dict = {
                    ckt.input_data: x,
                    ckt.input_skill: xx,
                    ckt.l: l,
                    ckt.next_id: next_id,
                    ckt.target_id: target_id,
                    ckt.target_correctness: target_correctness,
                    ckt.target_id2: target_id2,
                    ckt.target_correctness2: target_correctness2,
                    ckt.dropout_keep_prob: 1.0,
                    ckt.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [ckt.global_step, validation_summary_op, ckt.pred, ckt.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                return pred
            # Training loop. For each batch...
            
            run_time = []
            m_rmse = 1
            m_r2 = 0
            m_acc = 0
            m_auc = 0
            for iii in range(FLAGS.epochs):
                random.shuffle(train_students)
                a=datetime.now()
                data_size = len(train_students)
                index = 0
                actual_labels = []
                pred_labels = []
                while(index+FLAGS.batch_size < data_size):
                    x = np.zeros((FLAGS.batch_size, max_num_steps))
                    xx = np.zeros((FLAGS.batch_size, max_num_steps))
                    next_id = np.zeros((FLAGS.batch_size, max_num_steps))
                    l = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills))
                    target_id = []
                    target_correctness = []
                    target_id2 = []
                    target_correctness2 = []
                    for i in range(FLAGS.batch_size):
                        student = train_students[index+i]
                        problem_ids = student[1]
                        correctness = student[2]
                        correct_num = np.zeros(max_num_skills)
                        answer_count = np.ones(max_num_skills)
                        for j in range(len(problem_ids)-1):
                            problem_id = int(problem_ids[j])
                            
                            if(int(correctness[j]) == 0):
                                x[i, j] = problem_id + max_num_skills
                            else:
                                x[i, j] = problem_id
                                correct_num[problem_id] += 1
                            l[i,j] = correct_num / answer_count
                            answer_count[problem_id] += 1
                            xx[i,j] = problem_id
                            next_id[i,j] = int(problem_ids[j+1])

                            target_id.append(i*max_num_steps+j)
                            target_correctness.append(int(correctness[j+1]))
                            actual_labels.append(int(correctness[j+1]))
                        target_id2.append(i*max_num_steps+j)
                        target_correctness2.append(int(correctness[j+1]))
                        
                    index += FLAGS.batch_size
                    #print(ability)
                    pred = train_step(x, xx, l, next_id, target_id, target_correctness, target_id2, target_correctness2)
                    for p in pred:
                        pred_labels.append(p)
                    current_step = tf.train.global_step(sess, ckt.global_step)
                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)
                rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                auc = metrics.roc_auc_score(actual_labels, pred_labels)
                #calculate r^2
                r2 = r2_score(actual_labels, pred_labels)
                pred_score = np.greater_equal(pred_labels,0.5) 
                pred_score = pred_score.astype(int)
                pred_score = np.equal(actual_labels, pred_score)
                acc = np.mean(pred_score.astype(int))
                logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc{4:g} ".format((iii +1),rmse, auc, r2, acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")
                    
                    data_size = len(test_students)
                    index = 0
                    actual_labels = []
                    pred_labels = []
                    while(index+FLAGS.batch_size < data_size):
                        x = np.zeros((FLAGS.batch_size, max_num_steps))
                        xx = np.zeros((FLAGS.batch_size, max_num_steps))
                        next_id = np.zeros((FLAGS.batch_size, max_num_steps))
                        l = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills))
                        target_id = []
                        target_correctness = []
                        target_id2 = []
                        target_correctness2 = []
                        for i in range(FLAGS.batch_size):
                            student = test_students[index+i]
                            problem_ids = student[1]
                            correctness = student[2]
                            correct_num = np.zeros(max_num_skills)
                            answer_count = np.ones(max_num_skills)
                            for j in range(len(problem_ids)-1):
                                problem_id = int(problem_ids[j])
                                
                                if(int(correctness[j]) == 0):
                                    x[i, j] = problem_id + max_num_skills
                                else:
                                    x[i, j] = problem_id
                                    correct_num[problem_id] += 1
                                l[i,j] = correct_num / answer_count
                                answer_count[problem_id] += 1
                                xx[i,j] = problem_id
                                next_id[i,j] = int(problem_ids[j+1])
                                target_id.append(i*max_num_steps+j)
                                target_correctness.append(int(correctness[j+1]))
                                actual_labels.append(int(correctness[j+1]))
                            target_id2.append(i*max_num_steps+j)
                            target_correctness2.append(int(correctness[j+1]))
                            

                        index += FLAGS.batch_size
                        pred  = validation_step(x, xx, l, next_id, target_id, target_correctness, target_id2, target_correctness2)
                        for p in pred:
                                pred_labels.append(p)

                    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                    auc = metrics.roc_auc_score(actual_labels, pred_labels)
                    #calculate r^2
                    r2 = r2_score(actual_labels, pred_labels)
                    pred_score = np.greater_equal(pred_labels,0.5) 
                    pred_score = pred_score.astype(int)
                    pred_score = np.equal(actual_labels, pred_score)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: rmse {1:g}  auc {2:g}  r2 {3:g}   acc{4:g} ".format((iii +1)/FLAGS.evaluation_interval,rmse, auc, r2, acc))

                    if rmse < m_rmse:
                        m_rmse = rmse
                    if auc > m_auc:
                        m_auc = auc
                    if acc > m_acc:
                        m_acc = acc
                    if r2 > m_r2:
                        m_r2 = r2

                    best_saver.handle(auc, sess, current_step)
                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))
            
            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))
            logger.info("max: rmse {0:g}  auc {1:g}  r2 {2:g}   acc{3:g} ".format(m_rmse, m_auc, m_r2, m_acc))
    logger.info("Done.")


if __name__ == '__main__':
    train()
