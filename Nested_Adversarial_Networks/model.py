import layers as L 
import tensorflow as tf
import copy
import numpy as np 

acc = -1

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

def loadSess(modelpath=None,sess=None,modpath=None,mods=None,var_list=None,init=False):
#load session if there exist any models, and initialize the sess if not
	assert modpath==None or mods==None
	assert (not modelpath==None) or (not modpath==None) or (not modpath==None)
	if sess==None:
		sess = tf.Session()
	if init:
		sess.run(tf.global_variables_initializer())
	if var_list==None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list)
	
	if modpath!=None:
		mod = modpath
		print('loading from model:',mod)
		saver.restore(sess,mod)
	elif mods!=None:
		for m in mods:
			print('loading from model:',m)
			saver.restore(sess,m)
	elif modelpath!=None:
		ckpt = tf.train.get_checkpoint_state(modelpath)
		if ckpt:
			mod = ckpt.model_checkpoint_path
			print('loading from model:',mod)
			saver.restore(sess,mod)
		else:
			sess.run(tf.global_variables_initializer())
			print('No checkpoint in folder, use initial graph...')
	return sess

def initialize(sess):
	sess.run(tf.global_variables_initializer())

def get_feed_dict(keylist,vallist):
	assert len(keylist)==len(vallist)
	d = {}
	for i in range(len(keylist)):
		# print(keylist[i],'\t',type(vallist))
		d[keylist[i]] = vallist[i]
	return d

def get_trainable_vars(scope=None):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

def get_all_vars(scope=None):
	return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)

def get_update_ops(scope=None):
	return tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=scope)

class Model():
	def __init__(self,inp,size=None):
		self.result = inp
		if size is None:
			self.inpsize = inp.get_shape().as_list()
		else:
			self.inpsize = list(size)
		self.layernum = 0
		self.bntraining = True
		self.epsilon = None

	def set_bn_training(self,training):
		self.bntraining = training

	def set_bn_epsilon(self,epsilon):
		self.epsilon = epsilon

	def get_current_layer(self):
		return self.result

	def get_shape(self):
		return self.inpsize

	def activation(self,param):
		return self.activate(param)

	def activate(self,param):
		inp = self.result
		with tf.name_scope('activation_'+str(self.layernum)):
			if param == 0:
				res =  L.relu(inp,name='relu_'+str(self.layernum))
			elif param == 1:
				res =  L.lrelu(inp,name='lrelu_'+str(self.layernum))
			elif param == 2:
				res =  L.elu(inp,name='elu_'+str(self.layernum))
			elif param == 3:
				res =  L.tanh(inp,name='tanh_'+str(self.layernum))
			elif param == 4:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFM(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 5:
				self.inpsize[-1] = self.inpsize[-1]//2
				res =  L.MFMfc(inp,self.inpsize[-1],name='mfm_'+str(self.layernum))
			elif param == 6:
				res =  L.sigmoid(inp,name='sigmoid_'+str(self.layernum))
			else:
				res =  inp
		self.result = res
		return self.result

	def convLayer(self,size,outchn,dilation_rate=1,stride=1,pad='SAME',activation=-1,batch_norm=False,layerin=None,usebias=True,kernel_data=None,bias_data=None):
		with tf.variable_scope('conv_'+str(self.layernum)):
			if isinstance(size,list):
				kernel = size
			else:
				kernel = [size,size]
			if layerin!=None:
				self.result = layerin
				self.inpsize = layerin.get_shape().as_list()
			self.result = L.conv2D(self.result,kernel,outchn,'conv_'+str(self.layernum),stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.layernum += 1
			self.inpsize = self.result.get_shape().as_list()
			self.activate(activation)
		return self.result

	def maxpoolLayer(self,size,stride=None,pad='SAME'):
		if stride==None:
			stride = size
		self.result = L.maxpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def avgpoolLayer(self,size,stride=None,pad='SAME'):
		if stride==None:
			stride = size
		self.result = L.avgpooling(self.result,size,stride,'maxpool_'+str(self.layernum),pad=pad)
		self.inpsize = self.result.get_shape().as_list()
		return self.result

	def fcLayer(self,outsize,activation=-1,nobias=False,batch_norm=False):
		with tf.variable_scope('fc_'+str(self.layernum)):
			self.inpsize = [i for i in self.inpsize]
			self.result = L.Fcnn(self.result,self.inpsize[1],outsize,'fc_'+str(self.layernum),nobias=nobias)
			if batch_norm:
				self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
			self.inpsize[1] = outsize
			self.activate(activation)
			self.layernum+=1
		return self.result

	def concat_to_current(self,layerin,axis=3):
		with tf.variable_scope('concat'+str(self.layernum)):
			self.result = tf.concat(axis=axis,values=[self.result,layerin])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def concat_to_all_batch(self,layerinfo):
		with tf.variable_scope('concat'+str(self.layernum)):
			layerin = layerinfo
			layerin = tf.expand_dims(layerin,0)
			layerin = tf.tile(layerin,[tf.shape(self.result)[0],1,1,1])
			self.result = tf.concat(axis=-1,values=[self.result,layerin])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def set_current(self,layerinfo):
		if isinstance(layerinfo,list):
			self.result = layerinfo[0]
			self.inpsize = layerinfo[1]
		else:
			self.result = layerinfo
			self.inpsize = self.result.get_shape().as_list()

	def batch_norm(self):
		with tf.variable_scope('batch_norm'+str(self.layernum)):
			self.result = L.batch_norm(self.result,'batch_norm_'+str(self.layernum),training=self.bntraining,epsilon=self.epsilon)
		return self.result

	def reshape(self,shape):
		with tf.variable_scope('reshape_'+str(self.layernum)):
			self.result = tf.reshape(self.result,shape)
			self.inpsize = shape
		return self.result

	def sum(self,layerin):
		with tf.variable_scope('sum_'+str(self.layernum)):
			self.result = self.result +	layerin
		return self.result

	def pad(self,padding):
		with tf.variable_scope('pad_'+str(self.layernum)):
			# left, right, top, btm
			if isinstance(padding,list):
				self.result = tf.pad(self.result,[[0,0],[padding[0],padding[1]],[padding[2],padding[3]],[0,0]])
			else:
				self.result = tf.pad(self.result,[[0,0],[padding,padding],[padding,padding],[0,0]])
			self.inpsize = self.result.get_shape().as_list()
		return self.result

	def flatten(self):
		self.result = tf.reshape(self.result,[-1,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]])
		self.transShape = [self.inpsize[1],self.inpsize[2],self.inpsize[3],0]
		self.inpsize = [None,self.inpsize[1]*self.inpsize[2]*self.inpsize[3]]
		return self.result