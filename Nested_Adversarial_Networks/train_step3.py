import scipy.io as sio 
import tensorflow as tf
import model as M
import numpy as np
import img_reader
from widmod2 import seg_main_body
import random
from PIL import Image
import cv2
import data_provider

import os 
if not os.path.exists('./savings_inst_model/'):
    os.mkdir('./savings_inst_model/')

class network():
    def __init__(self, class_num):
        self.size = 460
        self.class_num = 20

        # build placeholders
        inp_holder = tf.placeholder(tf.float32,[None,size,size,3],name='image_holder')
        seg_holder = tf.placeholder(tf.float32,[None,size,size,class_num],name='segment_holder')
        mask_holder = tf.placeholder(tf.float32,[None,size,size],name='mask_holder')
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

        # build saver and session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # self.writer = tf.summary.FileWriter('./logs/',self.sess.graph)
        M.loadSess('./savings_inst_model/',self.sess,init=True,var_list=M.get_trainable_vars('inst_part/WideRes'))

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

        var_m = M.get_trainable_vars('inst_part/MergingLayer')
        var_c = M.get_trainable_vars('inst_part/stream')
        var_i = M.get_trainable_vars('inst_part/inst_layer')
        var_net = M.get_trainable_vars('inst_part/WideRes')

        with tf.variable_scope('overall_loss'):
            overall_loss = 5*coord_loss + inst_loss

        train_step = tf.train.AdamOptimizer(0.0001).minimize(overall_loss,var_list=var_m) # merging layer
        train_step2 = tf.train.AdamOptimizer(1e-5).minimize(overall_loss,var_list=var_net) # main body
        train_step4 = tf.train.AdamOptimizer(0.0001).minimize(10*coord_loss,var_list=var_c) # coord streams
        train_step5 = tf.train.AdamOptimizer(0.0001).minimize(inst_loss,var_list=var_i) # instant prediction
        with tf.control_dependencies([train_step,train_step2,train_step4,train_step5]):
            train_op = tf.no_op()

        self.crd_loss = coord_loss
        self.inst_loss = inst_loss
        self.train_op = train_op

    def train(self,img_batch,lab_batch,mask_batch,coord_batch,inst_batch):
        feed_d = {self.inp_holder:img_batch, self.lab_holder:lab_batch, self.mask_holder:mask_batch, self.coord_holder: coord_batch, self.inst_holder: inst_batch}
        seg_loss, crd_loss, inst_loss,_ = self.sess.run([self.seg_loss,self.crd_loss,self.inst_loss,self.train_op],feed_dict=feed_d)
        return seg_loss, crd_loss, inst_loss

    def save(self,path):
        self.saver.save(self.sess,path)

    def load(self,path):
        M.loadSess(sess=self.sess,modpath=path)

    def eval(self,img_batch,mask_batch):
        res = self.sess.run(self.seg_layer,feed_dict={self.inp_holder:img_batch, self.mask_holder:mask_batch})
        res = np.squeeze(res)
        res = np.argmax(res,-1)
        return res

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

def read_label(path):
    im = Image.open(path)
    img = np.array(im,np.uint8)
    return img

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b,g,r])

    cmap = cmap/255 if normalized else cmap
    return cmap

# Training

reader = data_reader('train.lst')
with tf.variable_scope('inst_part'):
    net = network()

MAX_ITER = 100000
BSIZE = 1
for i in range(MAX_ITER):
    img_batch, lab_batch, mask_batch, coord_batch, instnum_batch = reader.next_batch(BSIZE)
    seg_loss, coord_loss, inst_loss = net.train(img_batch,lab_batch,mask_batch,coord_batch,instnum_batch)
    print('Iter:%d\tSegLoss:%.4f\tCrdLoss:%.4f\tInstLoss:%.4f'%(i,seg_loss,coord_loss,inst_loss))

    # save the model
    if i%1000==0 and i>0:
        net.save('./savings_inst_model/%d.ckpt'%(i))
