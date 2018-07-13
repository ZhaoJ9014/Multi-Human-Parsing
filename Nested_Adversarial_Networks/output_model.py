import tensorflow as tf
import model as M
import numpy as np
import img_reader
from widmod2 import seg_main_body
import scipy.io as sio
import random
from PIL import Image
import cv2
import util

import os 
if not os.path.exists('./savings_overall/'):
    os.mkdir('./savings_overall/')

class network_bg_fg():
    def __init__(self,img_holder):
        inp_holder = img_holder

        self.net_body = seg_main_body(inp_holder)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12)

        self.inp_holder = inp_holder
        self.seg_layer = seg_layer

    def segmentation_layer(self,feature,dilation):
        with tf.variable_scope('SegLayer'):
            mod = M.Model(feature)
            mod.convLayer(3,2,dilation_rate=dilation)
        return mod.get_current_layer()

class network_seg():
    def __init__(self,img_holder,class_num,mask_layer):

        inp_holder = img_holder

        mask = tf.expand_dims(mask_layer,-1)
        c_ = tf.concat([inp_holder,mask],-1)
        merged_layer = self.merging_layer(c_)

        self.net_body = seg_main_body(merged_layer)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12,class_num)

        self.inp_holder = inp_holder
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

class network_inst():
    def __init__(self,img_holder, class_num, mask_layer, seg_layer):
        self.class_num = class_num

        # build placeholders
        inp_holder = img_holder

        # construct input (4 -> 3 with 1x1 conv)
        merged_layer = self.merging_layer(inp_holder,seg_layer,mask_layer)

        # build network
        self.get_coord(460)
        self.net_body = seg_main_body(merged_layer)

        stream_list = self.get_stream_list(self.net_body.feature_maps)
        inst_pred = self.inst_layer(self.net_body.feature_layer,stream_list[-1],class_num)

        # set class variables
        # holders
        self.inp_holder = inp_holder
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


class network():
    def __init__(self,class_num):
        img_holder = tf.placeholder(tf.float32,[None,460,460,3])
        with tf.variable_scope('bg_fg'):
            net_bgfg = network_bg_fg(img_holder)
        with tf.variable_scope('seg_part'):
            bg_fg_upsample = tf.nn.softmax(tf.image.resize_images(net_bgfg.seg_layer,img_holder.get_shape().as_list()[1:3]),1)[:,:,:,1]
            print(bg_fg_upsample)
            input('pause')
            net_seg = network_seg(img_holder,class_num,bg_fg_upsample)
        with tf.variable_scope('inst_part'):
            net_inst = network_inst(img_holder,class_num,
                tf.nn.softmax(tf.image.resize_images(net_bgfg.seg_layer,tf.shape(img_holder)[1:3]),1)[:,:,:,1],
                tf.image.resize_images(net_seg.seg_layer,tf.shape(img_holder)[1:3]))
        
        self.net_bgfg = net_bgfg
        self.net_seg = net_seg
        self.net_inst = net_inst

        self.img_holder = img_holder

        self.mask_out = tf.image.resize_images(net_bgfg.seg_layer,tf.shape(img_holder)[1:3])
        self.seg_out = tf.image.resize_images(net_seg.seg_layer,tf.shape(img_holder)[1:3])
        self.inst_num_out = net_inst.inst_layer
        self.coord_out = net_inst.coord_layer

        self.sess = tf.Session()
        M.loadSess('./savings_bgfg/',sess=self.sess,init=True,var_list=M.get_all_vars('bg_fg'))
        M.loadSess('./savings_seg/',sess=self.sess,var_list=M.get_all_vars('seg_part'))
        M.loadSess('./savings_inst/',sess=self.sess,var_list=M.get_all_vars('inst_part'))

    def eval(self,img_path):
        img = img_reader.read_img(img_path, None, scale=False)
        msk,seg,inst_num,coord = self.sess.run([self.mask_out, self.seg_out, self.inst_num_out, self.coord_out], feed_dict={self.img_holder: [img]})
        msk,seg,inst = util.get_output_img(msk[0],seg[0],inst_num[0],coord[0])
        return msk ,seg, inst