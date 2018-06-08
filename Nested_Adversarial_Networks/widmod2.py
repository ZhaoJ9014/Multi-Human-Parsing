import tensorflow as tf 
import numpy as np 
import model as M 

class seg_main_body():

	def conv_blocks(self,mod,featuremap,kernel=3,stride=2):
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		buff = mod.get_current_layer()
		mod.pad(1)
		mod.convLayer(kernel,featuremap,stride=stride,pad='VALID',activation=M.PARAM_RELU,batch_norm=True,usebias=False)
		mod.pad(1)
		branch = mod.convLayer(kernel,featuremap,pad='VALID',usebias=False)
		mod.set_current(buff)
		a = mod.convLayer(1,featuremap,stride=stride,usebias=False)
		mod.sum(branch)

	def identity_block(self,mod,featuremap,kernel=3,stride=1):
		buff = mod.get_current_layer()
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		mod.pad(1)
		mod.convLayer(kernel,featuremap,stride=stride,pad='VALID',activation=M.PARAM_RELU,batch_norm=True,usebias=False)
		mod.pad(1)
		mod.convLayer(kernel,featuremap,pad='VALID',usebias=False)
		mod.sum(buff)

	def block5(self,mod):
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		buff = mod.get_current_layer()
		mod.pad(1)
		mod.convLayer(3,512,pad='VALID',activation=M.PARAM_RELU,batch_norm=True,usebias=False)
		mod.pad(2)
		branch = mod.convLayer(3,1024,pad='VALID',usebias=False,dilation_rate=2)
		mod.set_current(buff)
		mod.convLayer(1,1024,usebias=False)
		mod.sum(branch)

	def id5_block(self,mod):
		buff = mod.get_current_layer()
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		mod.pad(2)
		mod.convLayer(3,512,pad='VALID',usebias=False,dilation_rate=2,batch_norm=True,activation=M.PARAM_RELU)
		mod.pad(2)
		mod.convLayer(3,1024,pad='VALID',usebias=False,dilation_rate=2)
		mod.sum(buff)

	def last_block(self,mod,featuremap,dilation_rate):
		mod.batch_norm()
		mod.activate(M.PARAM_RELU)
		buff = mod.get_current_layer()
		mod.convLayer(1,featuremap[0],pad='VALID',usebias=False,batch_norm=True,activation=M.PARAM_RELU)
		mod.pad(dilation_rate)
		mod.convLayer(3,featuremap[1],pad='VALID',usebias=False,batch_norm=True,activation=M.PARAM_RELU,dilation_rate=dilation_rate)
		branch = mod.convLayer(1,featuremap[2],pad='VALID',usebias=False)
		mod.set_current(buff)
		mod.convLayer(1,featuremap[2],pad='VALID',usebias=False)
		mod.sum(branch)

	def build_model(self,inpholder):
		with tf.variable_scope('WideRes'):
			mod = M.Model(inpholder)
			mod.set_bn_training(False)
			mod.set_bn_epsilon(1.0009999641624745e-05)
			mod.convLayer(3,64,usebias=False)
			c0 = mod.get_current_layer()
			self.conv_blocks(mod,128)
			self.identity_block(mod,128)
			self.identity_block(mod,128)
			c1 = mod.get_current_layer()
			self.conv_blocks(mod,256)
			self.identity_block(mod,256)
			self.identity_block(mod,256)
			c2 = mod.get_current_layer()
			self.conv_blocks(mod,512)
			self.identity_block(mod,512)
			self.identity_block(mod,512)
			self.identity_block(mod,512)
			self.identity_block(mod,512)
			self.identity_block(mod,512)
			c3 = mod.get_current_layer()
			self.block5(mod)
			self.id5_block(mod)
			self.id5_block(mod)
			self.last_block(mod,[512,1024,2048],4)
			self.last_block(mod,[1024,2048,4096],4)
			mod.batch_norm()
			mod.activate(M.PARAM_RELU)
			mod.pad(12)
			mod.convLayer(3,512,activation=M.PARAM_RELU,dilation_rate=12,pad='VALID')
		return c0,c1,c2,c3,mod.get_current_layer()

	def __init__(self,inp_holder):
		self.feature_maps = self.build_model(inp_holder)
		self.feature_layer = self.feature_maps[-1]
		self.var = M.get_all_vars('WideRes')