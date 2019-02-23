# Rework of model.py
# https://github.com/ddddwee1/sul
# This wrap-up is targeted for better touching low-level implementations 
import layers2 as L 
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config=config)
import numpy as np 
import os 
import random
import time

PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6

######## util functions ###########
def accuracy(pred,y,name='acc', one_hot=True):
	with tf.variable_scope(name):
		if one_hot:
			correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(tf.argmax(y,-1),tf.int64))
		else:
			correct = tf.equal(tf.cast(tf.argmax(pred,-1),tf.int64),tf.cast(y,tf.int64))
		acc = tf.reduce_mean(tf.cast(correct,tf.float32))
	return acc

##########################
# ETA class. I want to see the ETA. It's too boring to wait here.
class ETA():
	def __init__(self,max_value):
		self.start_time = time.time()
		self.max_value = max_value
		self.current = 0

	def start(self):
		self.start_time = time.time()
		self.current = 0

	def sec2hms(self,sec):
		hm = sec//60
		s = sec%60
		h = hm//60
		m = hm%60
		return h,m,s

	def get_ETA(self,current,is_string=True):
		self.current = current
		time_div = time.time() - self.start_time
		time_remain = time_div * float(self.max_value - self.current) / float(self.current + 1)
		h,m,s = self.sec2hms(int(time_remain))
		if is_string:
			return '%d:%d:%d'%(h,m,s)
		else:
			return h,m,s

########### universal model class ##########
class Model(tf.contrib.checkpoint.Checkpointable):
	def __init__(self,*args,**kwargs):
		self.initialized = False
		self.variables = []
		self.initialize(*args,**kwargs)

	def initialize(self,*args,**kwargs):
		pass

	def _gather_variables(self):
		self.variables = []
		atrs = dir(self)
		for i in atrs:
			if i[0] == '_':
				continue
			obj = getattr(self, i)
			self.variables += self._gather_variables_recursive(obj)

	def _gather_variables_recursive(self, obj):
		result = []
		if isinstance(obj, list) or isinstance(obj, tuple):
			for sub_obj in obj:
				result += self._gather_variables_recursive(sub_obj)
		elif isinstance(obj, Model) or isinstance(obj, L.Layer):
			result += obj.variables
		return result

	def get_variables(self, layers=None):
		if layers is None:
			return self.variables
		else:
			res = []
			for l in layers:
				res += l.variables
			return res 

	def set_bn_training(self, is_training):
		atrs = dir(self)
		# print(atrs)
		for i in atrs:
			if i[0] == '_':
				continue
			obj = getattr(self, i)
			self._set_bn_training_recursive(obj, is_training)

	def _set_bn_training_recursive(self, obj, is_training):
		if isinstance(obj, list):
			for sub_obj in obj:
				self._set_bn_training_recursive(sub_obj, is_training)
		if isinstance(obj, Model) and obj!=self:
			obj.set_bn_training(is_training)
		if isinstance(obj, L.batch_norm):
			obj.is_training = is_training

	def set_bn_epsilon(self, epsilon):
		atrs = dir(self)
		# print(atrs)
		for i in atrs:
			if i[0] == '_':
				continue
			obj = getattr(self, i)
			self._set_bn_epsilon_recursive(obj, epsilon)

	def _set_bn_epsilon_recursive(self, obj, epsilon):
		if isinstance(obj, list):
			for sub_obj in obj:
				self._set_bn_training_recursive(sub_obj, epsilon)
		if isinstance(obj, Model) and obj!=self:
			obj.set_bn_training(epsilon)
		if isinstance(obj, L.batch_norm):
			obj.epsilon = epsilon

	def __call__(self, x, *args, **kwargs):
		x = tf.convert_to_tensor(x, preferred_dtype=tf.float32)
		res = self.forward(x, *args, **kwargs)
		if not self.initialized:
			self._gather_variables()
			self.initialized = True
		return res 

########### universal layer classes ##########
class ConvLayer(Model):
	def initialize(self, size, outchn, dilation_rate=1, stride=1,pad='SAME',activation=-1,batch_norm=False, usebias=True,kernel_data=None,bias_data=None,weight_norm=False):
		self.conv = L.conv2D(size,outchn,stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate,weight_norm=weight_norm)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)
	def forward(self,x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class ConvLayer1D(Model):
	def initialize(self, size, outchn, dilation_rate=1, stride=1,pad='SAME',activation=-1,batch_norm=False, usebias=True,kernel_data=None,bias_data=None,weight_norm=False):
		self.conv = L.conv1D(size,outchn,stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate,weight_norm=weight_norm)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)
	def forward(self,x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class ConvLayer3D(Model):
	def initialize(self, size, outchn, dilation_rate=1, stride=1,pad='SAME',activation=-1,batch_norm=False, usebias=True,kernel_data=None,bias_data=None,weight_norm=False):
		self.conv = L.conv3D(size,outchn,stride=stride,pad=pad,usebias=usebias,kernel_data=kernel_data,bias_data=bias_data,dilation_rate=dilation_rate,weight_norm=weight_norm)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)
	def forward(self,x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class DeconvLayer(Model):
	def initialize(self, size, outchn, activation=-1, stride=1, usebias=True, pad='SAME', batch_norm=False):
		self.deconv = L.deconv2D(size,outchn,stride=stride,usebias=usebias,pad=pad, name=None)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)

	def forward(self,x):
		x = self.deconv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class DeconvLayer3D(Model):
	def initialize(self, size, outchn, activation=-1, stride=1, usebias=True, pad='SAME', batch_norm=False):
		self.deconv = L.deconv3D(size,outchn,stride=stride,usebias=usebias,pad=pad, name=None)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)

	def forward(self,x):
		x = self.deconv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class Dense(Model):
	def initialize(self, outsize, usebias=True, batch_norm=False, activation=-1):
		self.fclayer = L.fcLayer(outsize,usebias=usebias)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)

	def forward(self,x):
		x = self.fclayer(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 

class GraphConvLayer(Model):
	def initialize(self, outsize, adj_mtx=None, adj_fn=None, usebias=True, activation=-1, batch_norm=False):
		self.GCL = L.graphConvLayer(outsize, adj_mtx=adj_mtx, adj_fn=adj_fn, usebias=usebias)
		self.batch_norm = batch_norm
		self.activation_ = activation
		if batch_norm:
			self.bn = L.batch_norm()
		if activation!=-1:
			self.activation = L.activation(activation)

	def forward(self, x):
		x = self.GCL(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation_!=-1:
			x = self.activation(x)
		return x 


flatten = L.flatten()
maxPool = L.maxpoolLayer
avgPool = L.avgpoolLayer

########### higher wrapped block ##########

class ResBlock(Model):
	def initialize(self, outchn, stride=1, ratio=4, activation=PARAM_RELU):
		self.outchn = outchn
		# self.stride = stride
		self.activ = L.activation(activation)
		self.bn = L.batch_norm()
		self.l1 = ConvLayer(1, outchn//ratio, activation=PARAM_RELU, batch_norm=True)
		self.l2 = ConvLayer(3, outchn//ratio, activation=PARAM_RELU, batch_norm=True, stride=stride)
		self.l3 = ConvLayer(1, outchn)
		self.shortcut_conv = ConvLayer(1, outchn, activation=PARAM_RELU, stride=stride)
		self.shortcut_pool = L.maxpoolLayer(stride)

	def forward(self, x):
		inshape = x.get_shape().as_list()[-1]
		if inshape==self.outchn:
			short = self.shortcut_pool(x)
		else:
			short = self.shortcut_conv(x)

		branch = self.bn(x)
		branch = self.activ(branch)
		branch = self.l1(branch)
		branch = self.l2(branch)
		branch = self.l3(branch)

		return branch + short

class Sequential(Model):
	def initialize(self, modules):
		self.modules = modules

	def forward(self, x):
		for m in self.modules:
			x = m(x)
		return x

########### saver ##########
class Saver():
	def __init__(self, model, optim=None):
		self.mod = model

		self.obj = tf.contrib.checkpoint.Checkpointable()
		self.obj.m = self.mod
		self.optim = optim 
		if optim is None:
			self.ckpt = tf.train.Checkpoint(model=self.obj, optimizer_step=tf.train.get_or_create_global_step())
		else:
			self.ckpt = tf.train.Checkpoint(optimizer=optim, model=self.obj, optimizer_step=tf.train.get_or_create_global_step())
	
	def save(self, path):
		print('Saving model to path:',path)
		head, tail = os.path.split(path)
		if not os.path.exists(head):
			os.makedirs(head)
		self.ckpt.save(path)
		print('Model saved to path:',path)

	def restore(self, path, ptype='folder'):
		print('Load from:', path)
		try:
			if ptype=='folder':
				last_ckpt = tf.train.latest_checkpoint(path)
				print('Checkpoint:', last_ckpt)
				if last_ckpt is None:
					print('No model found in checkpoint.')
					print('Model will auto-initialize after first iteration.')
				self.ckpt.restore(last_ckpt)
			else:
				self.ckpt.restore(path)
			print('Finish loading.')
		except Exception as e:
			print('Model restore failed, Exception:',e)
			print('Model will auto-initialize after first iteration.')

######### Gradient accumulator #########
class GradAccumulator():
	def __init__(self):
		self.steps = 0
		self.grads = []

	def accumulate(self, grads):
		if len(grads) == 0:
			self.grads = grads
		else:
			for old_g, new_g in zip(self.grads, grads):
				old_g.assign_add(new_g)
		self.steps += 1

	def get_gradient(self):
		res = [i/self.steps for i in self.grads]
		self.grads = []
		self.steps = 0
		return res

	def get_step(self):
		return self.steps

######### Data Reader Template (serial) ##########
class DataReaderSerial():
	def __init__(self, one_hot=None):
		self.data_pos = 0
		self.val_pos = 0
		self.data = []
		self.val = []
		self.one_hot = False
		if one_hot is not None:
			self.one_hot = True
			self.eye = np.eye(one_hot)
		self.load_data()
		
	def get_next_batch(self,BSIZE):
		if self.data_pos + BSIZE > len(self.data):
			random.shuffle(self.data)
			self.data_pos = 0
		batch = self.data[self.data_pos : self.data_pos+BSIZE]
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.data_pos += BSIZE
		return x,y

	def get_val_next_batch(self, BSIZE):
		if self.val_pos + BSIZE >= len(self.val):
			batch = self.val[self.val_pos:]
			random.shuffle(self.val)
			self.val_pos = 0
			is_end = True
		else:
			batch = self.data[self.data_pos : self.data_pos+BSIZE]
			is_end = False
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.val_pos += BSIZE
		return x,y, is_end

	def get_train_iter(self, BSIZE):
		return len(self.data)//BSIZE

	def get_val_iter(self, BSIZE):
		return len(self.val)//BSIZE + 1

class ListReader():
	def __init__(self, one_hot=None):
		self.data_pos = 0
		self.val_pos = 0
		self.data = []
		self.val = []
		self.one_hot = False
		if one_hot is not None:
			self.one_hot = True
			self.eye = np.eye(one_hot)
		self.load_data()
		
	def get_next_batch(self,BSIZE):
		if self.data_pos + BSIZE > len(self.data):
			random.shuffle(self.data)
			self.data_pos = 0
		batch = self.data[self.data_pos : self.data_pos+BSIZE]
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.data_pos += BSIZE

		x = [self.process_img(i) for i in x]
		return x,y

	def get_val_next_batch(self, BSIZE):
		if self.val_pos + BSIZE >= len(self.val):
			batch = self.val[self.val_pos:]
			random.shuffle(self.val)
			self.val_pos = 0
			is_end = True
		else:
			batch = self.data[self.data_pos : self.data_pos+BSIZE]
			is_end = False
		x = [i[0] for i in batch]
		y = [i[1] for i in batch]
		if self.one_hot:
			y = self.eye[np.array(y)]
		self.val_pos += BSIZE
		x = [self.process_img(i) for i in x]
		return x,y, is_end

	def get_train_iter(self, BSIZE):
		return len(self.data)//BSIZE

	def get_val_iter(self, BSIZE):
		return len(self.val)//BSIZE + 1

######### Data Reader Template (parallel) ##########
# multi-process to read data
class DataReader():
	def __init__(self, data, fn, batch_size, shuffle=False, random_sample=False, processes=2, post_fn=None):
		from multiprocessing import Pool
		self.pool = Pool(processes)
		print('Starting parallel data loader...')
		self.process_fn = fn
		self.data = data
		self.batch_size = batch_size
		self.position = batch_size
		self.post_fn = post_fn
		self.random_sample = random_sample
		self.shuffle = shuffle
		if shuffle:
			random.shuffle(self.data)
		self._start_p(self.data[:batch_size])

	def _start_p(self, data):
		self.ps = []
		for i in data:
			self.ps.append(self.pool.apply_async(self.process_fn, [i]))

	def get_next_batch(self):
		# print('call')
		# fetch data
		res = [i.get() for i in self.ps]

		# start new pre-fetch
		if self.random_sample:
			batch = random.sample(self.data, self.batch_size)
		else:
			if self.position + self.batch_size > len(self.data):
				self.position = 0
				if self.shuffle:
					random.shuffle(self.data)	
			batch = self.data[self.position:self.position+self.batch_size]
			self.position += self.batch_size
		
		self._start_p(batch)

		# post_process the data
		if self.post_fn is not None:
			res = self.post_fn(res)
		return res 


######### short-cut functions #########

gradient_reverse = L.gradient_reverse

def pad(x, pad):
	if isinstance(pad, list):
		x = tf.pad(x, [[0,0],[pad[0],pad[1]], [pad[2],pad[3]], [0,0]])
	else:
		x = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[0,0]])
	return x 

def pad3D(x, pad):
	if isinstance(pad, list):
		x = tf.pad(x, [[0,0],[pad[0],pad[1]], [pad[2],pad[3]], [pad[4], pad[5]], [0,0]])
	else:
		x = tf.pad(x, [[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]])
	return x 

def image_transform(x, H, out_shape=None, interpolation='NEAREST'):
	# Will produce error if not specify 'output_shape' in eager mode
	shape = x.get_shape().as_list()
	if out_shape is None:
		if len(shape)==4:
			out_shape = shape[1:3]
		else:
			out_shape = shape[:2]
	return tf.contrib.image.transform(x, H, interpolation=interpolation, output_shape=out_shape)
 
def zip_grad(grads, vars):
	assert len(grads)==len(vars)
	grads_1 = []
	vars_1 = []
	for i in range(len(grads)):
		if not grads[i] is None:
			grads_1.append(grads[i])
			vars_1.append(vars[i])
	assert len(grads_1)!=0
	return zip(grads_1, vars_1)

