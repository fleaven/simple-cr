import tensorflow as tf

NUM_LABELS=34
prob=0.3 #dropout


def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
					   initializer=tf.constant_initializer(0.0),
						dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg


def adjnt_differ(im, dims):
	if dims==0:
		im2 = tf.concat([im[0:1], im[:-1]], dims)
	elif dims==1:
		im2 = tf.concat([im[:, 0:1], im[:, :-1]], dims)
	elif dims==2:
		im2 = tf.concat([im[:, :, 0:1], im[:,:,:-1]], dims)
	elif dims==3:
		im2 = tf.concat([im[:, :, :, 0:1], im[:, :, :, -1]], dims)
	else:
		raise ValueError('Invalid adjnt_differ dims %d' % dims)
	diff = tf.nn.relu(im - im2)
	return tf.reduce_mean(diff, dims)

def SumDiff(data, axis):
	data1 = tf.expand_dims(data, axis+2)
	v_sum = tf.reduce_mean(data1, axis)
	h_sum = tf.reduce_mean(data1, axis+1)
	v_diff = adjnt_differ(data1, axis)
	h_diff = adjnt_differ(data1, axis+1)
	sd = tf.concat([v_sum, h_sum, v_diff, h_diff], axis+1)
	return sd, tf.reduce_sum(sd, axis, keep_dims=True)

# weight initialization
def weight_variable(shape):
	initial = tf.random.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x1(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

def basiccnn3(input_tensor,train,regularizer,channels):
	conv1_deep = 96
	conv2_deep = 128
	conv3_deep = 160
	conv4_deep = 256
	convs1_deep = 128
	convs2_deep = 160
	convs3_deep = 256
	fc1_num=2048

	with tf.name_scope('layer1-conv1'):
		w_conv1 = weight_variable([5, 5, channels, conv1_deep])
		b_conv1 = bias_variable([conv1_deep])
		h_conv1 = tf.nn.relu(conv2d(input_tensor, w_conv1) + b_conv1)

	with tf.name_scope("layer2-pool1"):
		h_pool1 = max_pool_2x2(h_conv1) #48

	with tf.name_scope("layer3-conv2"):
		w_conv2 = weight_variable([5, 5, conv1_deep, conv2_deep])
		b_conv2 = bias_variable([conv2_deep])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

	with tf.name_scope("layer4-pool2"):
		h_pool2 = max_pool_2x2(h_conv2) #24

	with tf.name_scope("layer5-conv3"):
		w_conv3 = weight_variable([5, 5, conv2_deep, conv3_deep])
		b_conv3 = bias_variable([conv3_deep])
		h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

	with tf.name_scope("layer6-pool3"):
		h_pool3 = max_pool_2x2(h_conv3) #12

	with tf.name_scope("layer7-conv4"):
		w_conv4 = weight_variable([5, 5, conv3_deep, conv4_deep])
		b_conv4 = bias_variable([conv4_deep])
		h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)

	with tf.name_scope("layer6-pool3"):
		h_pool4 = max_pool_2x2(h_conv4)  # 6

	with tf.name_scope("layer-ss0"): #48*4
		kV = tf.constant(100.)
		bV = tf.constant(30.)
		x_bin, x_total = SumDiff(tf.sigmoid((h_pool3 - bV) * kV), 1)  # 对比度最大化后处理

	pool_shape = h_pool4.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	x_shape = x_bin.get_shape().as_list()
	x_nodes = x_shape[1] * x_shape[2] * x_shape[3]
	reshaped = tf.concat([tf.reshape(h_pool4, [-1, nodes]), tf.reshape(x_total[:,0,0,:], [-1,conv3_deep])], 1)

	with tf.name_scope("layer7-fc1"):
		w_fc1 = weight_variable([nodes +conv3_deep, fc1_num])
		b_fc1 = bias_variable([fc1_num])
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(w_fc1))
		h_fc1 = tf.nn.relu(tf.matmul(reshaped, w_fc1) + b_fc1)
#        keep_prob = tf.placeholder("float")
		if (train):
			h_fc1_drop = tf.nn.dropout(h_fc1, prob)
		else:
			h_fc1_drop = h_fc1

	with tf.name_scope("layer9-fc2"):
		w_fc2 = weight_variable([fc1_num, NUM_LABELS])
		b_fc2 = bias_variable([NUM_LABELS])
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(w_fc2))

	return tf.matmul(h_fc1_drop, w_fc2) + b_fc2, x_total