import tensorflow as tf 
import numpy as np 

l_num = 0

###########################################################
#define weight and bias initialization

def weight(shape,dtype=None):
	return tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=dtype)

def bias(shape,value=0.1,dtype=None):
	return tf.get_variable('bias',shape,initializer=tf.constant_initializer(value),dtype=dtype)

###########################################################
#define basic layers

def conv2D(x,size,outchn,name=None,stride=1,pad='SAME',usebias=True,kernel_data=None,bias_data=None,dilation_rate=1):
	global l_num
	print('Conv_bias:',usebias)
	if name is None:
		name = 'conv_l_'+str(l_num)
		l_num+=1
	# with tf.variable_scope(name):
	if isinstance(size,list):
		kernel = size
	else:
		kernel = [size,size]
	if (not kernel_data is None) and (not bias_data is None):
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.constant_initializer(kernel_data),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(bias_data),name=name)
	else:
		z = tf.layers.conv2d(x, outchn, kernel, strides=(stride, stride), padding=pad,\
			dilation_rate=dilation_rate,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			use_bias=usebias,\
			bias_initializer=tf.constant_initializer(0.1),name=name)
	return z

def sum(x,y):
	return x+y

def deconv2D(x,size,outchn,name,stride=1,pad='SAME'):
	with tf.variable_scope(name):
		if isinstance(size,list):
			kernel = size
		else:
			kernel = [size,size]
		z = tf.layers.conv2d_transpose(x, outchn, [size, size], strides=(stride, stride), padding=pad,\
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
			bias_initializer=tf.constant_initializer(0.1))
		return z

def maxpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'maxpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def avgpooling(x,size,stride=None,name=None,pad='SAME'):
	global l_num
	if name is None:
		name = 'avgpooling_l_'+str(l_num)
		l_num+=1
	with tf.variable_scope(name):
		if stride is None:
			stride = size
		return tf.nn.avg_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding=pad)

def Fcnn(x,insize,outsize,name,activation=None,nobias=False,dtype=None):
	if dtype is None:
		dtype = x.dtype
	with tf.variable_scope(name):
		if nobias:
			print('No biased fully connected layer is used!')
			W = weight([insize,outsize],dtype=dtype)
			tf.summary.histogram(name+'/weight',W)
			if activation==None:
				return tf.matmul(x,W)
			return activation(tf.matmul(x,W))
		else:
			W = weight([insize,outsize],dtype=dtype)
			b = bias([outsize],dtype=dtype)
			tf.summary.histogram(name+'/weight',W)
			tf.summary.histogram(name+'/bias',b)
			if activation==None:
				return tf.matmul(x,W)+b
			return activation(tf.matmul(x,W)+b)

def MFM(x,half,name):
	with tf.variable_scope(name):
		#shape is in format [batchsize, x, y, channel]
		# shape = tf.shape(x)
		shape = x.get_shape().as_list()
		res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2])
		res = tf.reduce_max(res,axis=[3])
		return res

def MFMfc(x,half,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		# print('fcshape:',shape)
		res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),reduction_indices=[1])
	return res

def batch_norm(inp,name,epsilon=None,variance=None,training=True):
	print('BN training:',training)
	if not epsilon is None:
		return tf.layers.batch_normalization(inp,training=training,name=name,epsilon=epsilon)
	return tf.layers.batch_normalization(inp,training=training,name=name)

def lrelu(x,name,leaky=0.2):
	return tf.maximum(x,x*leaky,name=name)

def relu(inp,name):
	return tf.nn.relu(inp,name=name)

def tanh(inp,name):
	return tf.tanh(inp,name=name)

def elu(inp,name):
	return tf.nn.elu(inp,name=name)

def sigmoid(inp,name):
	return tf.sigmoid(inp,name=name)
