import tensorflow as tf
import model as M
import numpy as np
import img_reader
from widmod2 import seg_main_body
import scipy.io as sio
import random
from PIL import Image
import cv2

import os 
if not os.path.exists('./savings_overall/'):
    os.mkdir('./savings_overall/')

class network_bg_fg():
    def __init__(self,img_holder):
        inp_holder = img_holder
        lab_holder = tf.placeholder(tf.int32,[None,460,460])

        self.net_body = seg_main_body(inp_holder)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12)
        self.build_loss(seg_layer,lab_holder)

        self.inp_holder = inp_holder
        self.lab_holder = lab_holder
        self.seg_layer = seg_layer

    def segmentation_layer(self,feature,dilation):
        with tf.variable_scope('SegLayer'):
            mod = M.Model(feature)
            mod.convLayer(3,2,dilation_rate=dilation)
        return mod.get_current_layer()

    def build_loss(self,seg_layer,lab_holder):
        lab_reform = tf.expand_dims(lab_holder,-1)
        lab_reform = tf.image.resize_images(seg_layer,tf.shape(lab_reform)[1:3])
        lab_reform = tf.squeeze(lab_reform)
        seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_layer,labels=lab_reform))

        var_s = M.get_trainable_vars('SegLayer')

        self.loss = seg_loss

class network_seg():
    def __init__(self,img_holder,class_num,mask_layer):

        inp_holder = img_holder
        lab_holder = tf.placeholder(tf.int32,[None,460,460])

        mask = tf.expand_dims(mask_layer,-1)
        c_ = tf.concat([inp_holder,mask],-1)
        merged_layer = self.merging_layer(c_)

        self.net_body = seg_main_body(merged_layer)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12,class_num)
        self.build_loss(seg_layer,lab_holder)

        self.inp_holder = inp_holder
        self.lab_holder = lab_holder
        self.seg_layer = seg_layer
        self.mask_layer = mask_layer

    def merging_layer(self,inp):
        with tf.variable_scope('MergingLayer'):
            mod = M.Model(inp)
            mod.convLayer(1,3,usebias=False)
        return mod.get_current_layer()

    def segmentation_layer(self,feature,dilation,class_num):
        with tf.variable_scope('SegLayer'):
            mod = M.Model(feature)
            mod.convLayer(3,class_num,dilation_rate=dilation)
        return mod.get_current_layer()

    def build_loss(self,seg_layer,lab_holder):
        lab_reform = tf.expand_dims(lab_holder,-1)
        lab_reform = tf.image.resize_images(seg_layer,tf.shape(lab_reform)[1:3])
        lab_reform = tf.squeeze(lab_reform)
        seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_layer,labels=lab_reform))

        self.loss = seg_loss

class network_inst():
    def __init__(self,img_holder, class_num, mask_layer, seg_layer):
        self.size = 460
        self.class_num = class_num

        # build placeholders
        inp_holder = img_holder
        seg_holder = seg_layer
        mask_holder = mask_layer
        coord_holder = tf.placeholder(tf.float32,[None,size,size,6],name='coordinate_holder')
        inst_holder = tf.placeholder(tf.float32,[None,class_num],name='instance_holder')

        # construct input (4 -> 3 with 1x1 conv)
        merged_layer = self.merging_layer(inp_holder,seg_holder,mask_holder)

        # build network
        self.get_coord(size)
        self.net_body = seg_main_body(merged_layer)

        stream_list = self.get_stream_list(self.net_body.feature_maps)
        inst_pred = self.inst_layer(self.net_body.feature_layer,stream_list[-1],class_num)
        self.build_loss(seg_layer,stream_list,inst_pred,lab_holder,mask_holder,coord_holder,inst_holder)

        # set class variables
        # holders
        self.inp_holder = inp_holder
        self.lab_holder = lab_holder
        self.mask_holder = mask_holder
        self.coord_holder = coord_holder
        self.inst_holder = inst_holder
        # layers
        self.coord_layer = stream_list[-1]
        self.inst_layer = inst_pred

    def merging_layer(self,inp,seg,mask):
        with tf.variable_scope('MergingLayer'):
            mask = tf.expand_dims(mask, -1)
            c_ = tf.concat([inp, seg, mask], -1)
            mod = M.Model(c_)
            mod.convLayer(1,3,usebias=False)
        return mod.get_current_layer()

    def inst_layer(self,feature_layer,stream_last,class_num):
        with tf.variable_scope('inst_layer'):
            mod = M.Model(tf.concat([feature_layer,stream_last],-1))
            mod.convLayer(3,128,stride=2,activation=M.PARAM_RELU)
            mod.flatten()
            mod.fcLayer(1024,activation=M.PARAM_RELU)
            mod.fcLayer(class_num)
        return mod.get_current_layer()

    def get_coord(self,size):
        res = np.zeros([size,size,2],np.float32)
        for i in range(size):
            for j in range(size):
                res[i][j] = [i,j]
        self.coords = res

    def get_stream_list(self,feature_maps):
        stream_list = []
        with tf.variable_scope('stream'):
            for i,feature in enumerate(feature_maps):
                if i<3:
                    continue
                with tf.variable_scope('stream_%d'%(i)):
                    mod = M.Model(feature)
                    mod.batch_norm()
                    mod.convLayer(3,128,activation=M.PARAM_RELU,batch_norm=True)
                    mod.convLayer(3,128,activation=M.PARAM_RELU,batch_norm=True)
                    mod.convLayer(1,6)
                    stream_list.append(mod.get_current_layer())

            # get fusion predition
            with tf.variable_scope('stream_%d'%(i+1)):
                buff = []
                for stream in stream_list:
                    buff.append(tf.image.resize_images(stream,tf.shape(stream_list[-1])[1:3],tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                mod = M.Model(tf.concat(buff,-1))
                mod.convLayer(1,6)
                stream_list.append(mod.get_current_layer())
        return stream_list

    def stream_loss(self,stream,label,mask):
        mask = tf.expand_dims(mask,-1)
        mask = tf.tile(mask,[1,1,1,6])
        label = label * mask 
        label = tf.image.resize_images(label,tf.shape(stream)[1:3],tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.image.resize_images(mask,tf.shape(stream)[1:3],tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        c = tf.abs(label * mask - stream * mask)
        a = tf.reduce_sum(c)
        b = tf.reduce_sum(mask)
        loss = a/(b+1)
        self.debug_layer = label
        return loss

    def coord_loss(self,stream_list,coord,mask):
        loss = 0
        for i,stream in enumerate(stream_list):
            with tf.variable_scope('coord_loss_%d'%(i)):
                loss += self.stream_loss(stream,coord,mask)
        return loss

    def build_loss(self,seg_layer,stream_list,inst_layer,lab_holder,mask_holder,coord_holder,inst_holder):

        with tf.variable_scope('coord_loss'):
            coords_reshape = tf.constant(self.coords[:,:,::-1],tf.float32)
            coords_reshape = tf.expand_dims(coords_reshape,0)
            offset = coord_holder - tf.tile(coords_reshape,[tf.shape(coord_holder)[0],1,1,3])
            coord_loss = self.coord_loss(stream_list,offset,mask_holder)
        with tf.variable_scope('inst_loss'):
            inst_loss = tf.reduce_mean(tf.square(inst_layer - inst_holder))

        with tf.variable_scope('overall_loss'):
            overall_loss = 5*coord_loss + inst_loss

        self.crd_loss = coord_loss
        self.inst_loss = inst_loss
        self.overall_loss = overall_loss

class network():
	def __init__(self,class_num):
		img_holder = tf.placeholder(tf.float32,[None,460,460,3])
		with tf.variable_scope('bg_fg'):
			net_bgfg = network_bg_fg(img_holder)
		with tf.variable_scope('seg_part'):
			net_seg = network_seg(class_num,img_holder,tf.nn.softmax(tf.image.resize_images(net_bgfg.seg_layer,[460,460]),1)[:,:,:,1])
		with tf.variable_scope('inst_part'):
			net_inst = network_inst(class_num,img_holder
				tf.nn.softmax(tf.image.resize_images(net_bgfg.seg_layer,[460,460]),1)[:,:,:,1],
				tf.image.resize_images(net_seg.seg_layer,[460,460]))
		
		self.network_bg_fg = network_bg_fg
		self.network_seg = network_seg
		self.network_inst = network_inst

		self.img_holder = img_holder
		self.mask_holder = network_bg_fg.lab_holder
		self.seg_holder = network_seg.lab_holder
		self.coord_holder = network_inst.coord_holder
		self.inst_holder = network_inst.inst_holder

		self.mask_out = network_bg_fg.seg_layer
		self.seg_out = network_seg.seg_layer
		self.inst_num_out = network_inst.inst_layer
		self.coord_out = network_inst.coord_layer

		self.build_loss()

		self.sess = tf.Session()
		M.loadSess('./savings_bgfg/',sess=self.sess,init=True,var_list=M.get_all_vars('bg_fg'))
		M.loadSess('./savings_seg/',sess=self.sess,var_list=M.get_all_vars('seg_part'))
		M.loadSess('./savings_inst/',sess=self.sess,var_list=M.get_all_vars('inst_part'))

	def build_loss(self):
		with tf.variable_scope('mask_loss'):
			lab_reform = tf.expand_dims(self.mask_holder,-1)
			lab_reform = tf.image.resize_images(seg_layer,tf.shape(lab_reform)[1:3])
			lab_reform = tf.squeeze(lab_reform)
			mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.mask_out,labels=lab_reform))
			train_mask0 = tf.train.AdamOptimizer(0.0001).minimize(mask_loss,var_list=M.get_trainable_vars('bg_fg/SegLayer'))
			train_mask1 = tf.train.AdamOptimizer(1e-5).minimize(mask_loss,var_list=M.get_trainable_vars('bg_fg/WideRes'))

		with tf.variable_scope('seg_loss'):
			lab_reform = tf.expand_dims(self.seg_holder,-1)
	        lab_reform = tf.image.resize_images(self.seg_out,tf.shape(lab_reform)[1:3])
	        lab_reform = tf.squeeze(lab_reform)
	        seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.seg_out,labels=lab_reform))
	        train_seg0 = tf.train.AdamOptimizer(0.0001).minimize(seg_loss,var_list=M.get_trainable_vars('seg_part/SegLayer')+M.get_trainable_vars('seg_part/MergingLayer'))
	        train_seg1 = tf.train.AdamOptimizer(1e-5).minimize(seg_loss,var_list=M.get_trainable_vars('seg_part/WideRes'))

	    with tf.variable_scope('inst_loss'):
	    	train_inst0 = tf.train.AdamOptimizer(0.0001).minimize(self.network_inst.overall_loss,var_list=M.get_trainable_vars('inst_part/MergingLayer')) # merging layer
	        train_inst1 = tf.train.AdamOptimizer(1e-5).minimize(self.network_inst.overall_loss,var_list=M.get_trainable_vars('inst_part/WideRes')) # main body
	        train_inst2 = tf.train.AdamOptimizer(0.0001).minimize(10*self.network_inst.coord_loss,var_list=M.get_trainable_vars('inst_part/stream')) # coord streams
	        train_inst3 = tf.train.AdamOptimizer(0.0001).minimize(self.network_inst.inst_loss,var_list=M.get_trainable_vars('inst_part/inst_layer')) # instant prediction

	    upd_ops = M.get_update_ops()

	    with tf.control_dependencies(upd_ops+[train_mask0,train_mask1,train_seg1,train_seg0,train_inst0,train_inst1,train_inst2,train_inst3]):
	    	train_op = tf.no_op()

	    self.mask_loss = mask_loss
	    self.seg_loss = seg_loss
	    self.inst_loss = self.network_inst.inst_loss
	    self.coord_loss = self.network_inst.coord_loss
	    self.train_op = train_op

	def train(self,img_batch,lab_batch,mask_batch,coord_batch,inst_batch):
		feed_d = {self.img_holder:img_batch, self.seg_holder:lab_batch, self.mask_holder:mask_batch, self.coord_holder: coord_batch, self.inst_holder: inst_batch}
		mask_loss, seg_loss, inst_loss, coord_loss,_ = self.sess.run([self.mask_loss,self.seg_loss,self.inst_loss,self.coord_loss,self.train_op],feed_dict=feed_d)
		return mask_loss, seg_loss, inst_loss, coord_loss

class data_reader():
    def __init__(self,data_file):
        print('Reading data...')
        self.fnames = []
        f = open(data_file)
        self.data = []
        cnt = 0
        for i in f:
            i = i.strip().split('\t')
            jpgfile = i[0]

            # get jpg, seg, mask
            jpg = img_reader.read_img(jpgfile,500,padding=True)
            seg = data_provider.get_seg(jpgfile)
            seg = img_reader.pad(seg,np.uint8,False)
            mask = seg.copy()
            seg[seg==255] = 0
            mask[mask!=0] = 1
            coords = data_provider.get_coord_map(jpgfile)
            coords = img_reader.pad(coords,np.float32,6)
            inst_num = data_provider.get_inst_num(jpgfile)

            if seg.shape[0]!=500:
                print(jpgfile)
                continue

            # if size is ok, add to data list
            self.data.append([jpg,seg,mask,coords,inst_num])
            self.fnames.append(jpgfile)
            cnt += 1
            if cnt%100==0:
                print(cnt)

    def next_batch(self,BSIZE):
        res = random.sample(self.data,BSIZE)
        img_batch = []
        lab_batch = []
        mask_batch = []
        coords_batch = []
        instnum_batch = []
        for i in range(BSIZE):
            img,lab,mask,coords,instnum = self.random_crop(res[i])
            img,lab,mask,coords,instnum = self.random_flip(img,lab,mask,coords,instnum)
            img_batch.append(img)
            lab_batch.append(lab)
            mask_batch.append(mask)
            coords_batch.append(coords*np.expand_dims(np.float32(mask),-1))
            instnum_batch.append(instnum)
        return img_batch, lab_batch, mask_batch,coords_batch, instnum_batch

    def random_crop(self,inp):
        img, lab, mask,coords,instnum = inp
        # print(coords[250,240:250])
        h = random.randint(0,40)
        w = random.randint(0,40)
        img = img[h:h+460,w:w+460]
        lab = lab[h:h+460,w:w+460]
        mask = mask[h:h+460,w:w+460]
        coords = coords[h:h+460,w:w+460]
        coords = coords - np.float32([[[h,w,h,w,h,w]]])
        return img,lab,mask,coords,instnum

    def random_flip(self,img,lab,mask,coords,instnum):
        if random.random()<0.5:
            coords_flip = coords.copy()
            coords_flip[:,:,0] = 460 - coords_flip[:,:,0]
            coords_flip[:,:,2] = 460 - coords_flip[:,:,2]
            coords_flip[:,:,4] = 460 - coords_flip[:,:,4]
            return img[:,::-1], lab[:,::-1], mask[:,::-1],coords_flip,instnum
        return img, lab, mask, coords, instnum

    def read_label(self,path):
        im = Image.open(path)
        img = np.array(im,np.uint8)
        return img


reader = data_provider('train.list')

net = network()

MAX_ITER = 100000
BSIZE = 1
for i in range(MAX_ITER):
    img_batch, lab_batch, mask_batch, coord_batch, instnum_batch = reader.next_batch(BSIZE)
    mask_loss, seg_loss, coord_loss, inst_loss = net.train(img_batch,lab_batch,mask_batch,coord_batch,instnum_batch)
    print('Iter:%d\tMaskLoss:%.4f\tSegLoss:%.4f\tCrdLoss:%.4f\tInstLoss:%.4f'%(i,mask_loss,seg_loss,coord_loss,inst_loss))

    # save the model
    if i%1000==0 and i>0:
        net.save('./savings_overall/%d.ckpt'%(i))