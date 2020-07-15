#-*- coding=utf-8 -*-
import os

import logging

import tensorflow as tf
import data
from models import hccr_cnnnet
from models import basiccnn
from models import basiccnn1
from models import basiccnn2
from models import basiccnn3

class tester:
    def __init__(self, savep, trainp,proc):
        self.save_path=savep #模型保存路径
        self.test_dir=trainp      #训练图片路径
        self.process=proc

x1 = tester('./checkpoint_x1', '../ocr-dataset/hwdb/minitest', basiccnn1)
x2 = tester('./checkpoint_x2', '../ocr-dataset/hwdb/minitest', basiccnn2)
x3 = tester('./checkpoint_x3', '../ocr-dataset/hwdb/minitest', basiccnn3)
basic = tester('./checkpoint_basic', '../ocr-dataset/hwdb/minitest', basiccnn)
hccr = tester('./checkpoint', '../ocr-dataset/hwdb/minitest', hccr_cnnnet)

tr = hccr

logger = logging.getLogger()
logger.setLevel(level = logging.INFO)
loghandler = logging.FileHandler(tr.save_path+"/test_logs.txt")
logger.addHandler(loghandler)
logger.addHandler(logging.StreamHandler())

gpunum='2'

batch_size = 128
img_size=[96,96]
channels=1

os.environ['CUDA_VISIBLE_DEVICES']=gpunum
with tf.Graph().as_default() as g:

    label_batch, image_batch = data.load_data(tr.test_dir, 0, batch_size, channels)
    
#    logits=hccr_cnnnet(image_batch,train=False,regularizer=None,channels=channels)
    logits,tt=tr.process(image_batch,train=False,regularizer=None,channels=channels)
    prob_batch = tf.nn.softmax(logits)
    accuracy_top1_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 1), tf.float32))
    accuracy_top5_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 2), tf.float32))
    accuracy_top10_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 3), tf.float32))
    '''
    variable_ave = tf.train.ExponentialMovingAverage(0.99)
    variables_to_restore = variable_ave.variables_to_restore()
    '''
    saver=tf.train.Saver()
    
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(tr.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            iternum=0
            top1sum=0
            top5sum=0
            top10sum=0

            while True:
                try:
                    top1,top5,top10 = sess.run([accuracy_top1_batch,accuracy_top5_batch,accuracy_top10_batch])
                    iternum=iternum+1
                    top1sum=top1sum+top1
                    top5sum=top5sum+top5
                    top10sum=top10sum+top10
                    if iternum%500==0:
                        logger.info("The current test accuracy (in %d pics) = top1: %g , top2: %g ，top3: %g." % (iternum*batch_size,top1sum/iternum,top5sum/iternum,top10sum/iternum))
                except tf.errors.OutOfRangeError:
                    logger.info("The final test accuracy (in %d pics) = top1: %g , top5: %g ，top10: %g." % (iternum*batch_size,top1sum/iternum,top5sum/iternum,top10sum/iternum))
                    logger.info('Test finished...')
                    break
        else:
            logger.info('No checkpoint file found !')
