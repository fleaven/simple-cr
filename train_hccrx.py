#-*- coding=utf-8 -*-
import os
from models import hccr_cnnnet
from models import basiccnn
from models import basiccnn1
from models import basiccnn2
from models import basiccnn3
from models import hccrx

class trainner:
    def __init__(self, savep, trainp,proc):
        self.save_path=savep #模型保存路径
        self.train_dir=trainp      #训练图片路径
        self.process=proc

x1 = trainner('./checkpoint_x1', '../ocr-dataset/hwdb/minitrain', basiccnn1)
x2 = trainner('./checkpoint_x2', '../ocr-dataset/hwdb/minitrain', basiccnn2)
x3 = trainner('./checkpoint_x3', '../ocr-dataset/hwdb/minitrain', basiccnn3)
basic = trainner('./checkpoint_basic', '../ocr-dataset/hwdb/minitrain', basiccnn)
hccr = trainner('./checkpoint', '../ocr-dataset/hwdb/minitrain', hccr_cnnnet)
hccrx = trainner('./checkpoint_hccrx', '../ocr-dataset/hwdb/minitrain', hccrx)

tr = hccrx

import tensorflow as tf
from signal import SIGINT, SIGTERM
import data
import lbtoolbox as lb
import logging

logger = logging.getLogger()
logger.setLevel(level = logging.DEBUG)
loghandler = logging.FileHandler(tr.save_path+"/train_logs.txt")
logger.addHandler(loghandler)
logger.addHandler(logging.StreamHandler())




gpunum='2'
lr_base=0.1
lr_decay=0.1
momentum=0.9
lr_steps=7000
save_steps=7000
print_steps=100
train_nums=30000
buffer_size=1000
regular_rate=0.0005

batch_size = 128
channels=1


resume=True #是否继续训练模型？


'''
losslist = []
accuracy = []
'''
os.environ['CUDA_VISIBLE_DEVICES']=gpunum


label_batch, image_batch = data.load_data(tr.train_dir, buffer_size, batch_size, channels)
regularizer=tf.contrib.layers.l2_regularizer(regular_rate)

logits, tt=tr.process(image_batch,train=True,regularizer=regularizer,channels=channels)

global_step=tf.Variable(0,trainable=False)

prob_batch = tf.nn.softmax(logits)
accuracy_top1_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 1), tf.float32))
accuracy_top5_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 2), tf.float32))
accuracy_top10_batch = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_batch, label_batch, 3), tf.float32))

update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

variable_ave = tf.train.ExponentialMovingAverage(0.99,global_step)
ave_op = variable_ave.apply(tf.trainable_variables())

cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
if regularizer==None:
    loss=cross_entropy_mean
else:
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

lr=tf.train.exponential_decay(lr_base,global_step,lr_steps,lr_decay,staircase=True)
train_step = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum)
with tf.control_dependencies(update_op):
    grads = train_step.compute_gradients(loss)
    train_op = train_step.apply_gradients(grads, global_step=global_step)

var_list = tf.trainable_variables()
if global_step is not None:
    var_list.append(global_step)
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    if resume:
            last_checkpoint = tf.train.latest_checkpoint(tr.save_path)
            saver.restore(sess, last_checkpoint)
            start_step = sess.run(global_step)
            logger.debug('Resume training ... Start from step %d / %d .'%(start_step,train_nums))
            resume=False
    else:
            start_step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
      for i in range(start_step,train_nums):

        _,loss_value,step, ttt=sess.run([train_op,loss,global_step, tt])
        if i % print_steps == 0:
            top1,top5,top10=sess.run([accuracy_top1_batch,accuracy_top5_batch,accuracy_top10_batch])
            logger.debug("After %d training step(s),loss on training batch is %g.The batch test accuracy = %g , %g ，%g."%(i,loss_value,top1,top5,top10))
            '''
            losslist.append([step,loss_value])
            accuracy.append([step,top1])
            '''
        if (i!=0 and i % save_steps == 0):
                    model_name="trainnum_%d_"%train_nums
                    saver.save(sess, os.path.join(tr.save_path, model_name), global_step=global_step)

        if u.interrupted:
                    logger.debug("Interrupted on request...")
                    break
                    
    '''              
    file1=open(log_dir+'/loss.txt','a')
    for loss in losslist:
          loss = str(loss).strip('[').strip(']').replace(',','')
          file1.write(loss+'\n')
    file1.close()
            
    file2=open(log_dir+'/accu.txt','a')
    for acc in accuracy:
          acc = str(acc).strip('[').strip(']').replace(',','')
          file2.write(acc+'\n')
    file2.close()
    '''

    logger.debug("----------------ss->bn_conv6")
    model_name="trainnum_%d_"%train_nums
    saver.save(sess,os.path.join(tr.save_path,model_name),global_step=global_step)
    logger.debug('Train finished...')
         
    coord.request_stop()
    coord.join(threads)
