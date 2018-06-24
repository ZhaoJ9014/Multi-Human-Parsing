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
if not os.path.exists('./savings_seg/'):
    os.mkdir('./savings_seg/')
if not os.path.exists('./sample_seg/'):
    os.mkdir('./sample_seg/')

class network():
    def __init__(self,class_num):

        inp_holder = tf.placeholder(tf.float32,[None,460,460,3])
        lab_holder = tf.placeholder(tf.int32,[None,460,460])
        mask_holder = tf.placeholder(tf.float32,[None,460,460])

        mask = tf.expand_dims(mask_holder,-1)
        c_ = tf.concat([inp_holder,mask],-1)
        merged_layer = self.merging_layer(c_)

        self.net_body = seg_main_body(merged_layer)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12,class_num)
        self.build_loss(seg_layer,lab_holder)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        M.loadSess('./savings_seg/',self.sess,init=True,var_list=M.get_trainable_vars('seg_part/WideRes'))

        self.inp_holder = inp_holder
        self.lab_holder = lab_holder
        self.seg_layer = seg_layer
        self.mask_holder = mask_holder

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

        var_s = M.get_trainable_vars('seg_part/SegLayer')
        var_m = M.get_trainable_vars('seg_part/MergingLayer')
        var_net = M.get_trainable_vars('seg_part/WideRes')

        train_step = tf.train.AdamOptimizer(0.0001).minimize(seg_loss,var_list=var_s)
        train_step2 = tf.train.AdamOptimizer(1e-5).minimize(seg_loss,var_list=var_net)
        train_step3 = tf.train.AdamOptimizer(0.0001).minimize(seg_loss,var_list=var_m)
        upds = M.get_update_ops()
        with tf.control_dependencies(upds+[train_step,train_step2,train_step3]):
            train_op = tf.no_op()

        self.loss = seg_loss
        self.train_op = train_op

    def train(self,img_batch,lab_batch,mask_batch):
        ls,_ = self.sess.run([self.loss,self.train_op],feed_dict={self.inp_holder:img_batch, self.lab_holder:lab_batch, self.mask_holder:mask_batch})
        return ls

    def save(self,path):
        self.saver.save(self.sess,path)

    def load(self,path):
        M.loadSess(sess=self.sess,modpath=path)

    def eval(self,img_batch,mask_batch):
        res = self.sess.run(self.seg_layer,feed_dict={self.inp_holder:img_batch, self.mask_holder:mask_batch})
        res = np.squeeze(res)
        res = np.argmax(res,-1)
        return res

class data_provider():
    def __init__(self,data_file):
        print('Reading data...')
        self.fnames = []
        f = open(data_file)
        self.data = []
        cnt = 0
        for i in f:
            i = i.strip()
            jpgfile = './data/reform_img/'+i
            segfile = './data/reform_annot/'+i.replace('.jpg','.png')
            jpg = img_reader.read_img(jpgfile,500,padding=True)
            seg = self.read_label(segfile)
            seg = img_reader.pad(seg,np.uint8,False)
            mask = seg.copy()
            seg[seg==255] = 0
            mask[mask!=0] = 1
            if seg.shape[0]!=500:
                print(segfile)
                continue
            self.data.append([jpg,seg,mask])
            self.fnames.append(jpgfile)
            cnt += 1
            if cnt%100==0:
                print(cnt)
        print('Data length:',len(self.data))

    def next_batch(self,BSIZE):
        res = random.sample(self.data,BSIZE)
        img_batch = []
        lab_batch = []
        mask_batch = []
        for i in range(BSIZE):
            img,lab,mask = self.random_crop(res[i])
            img,lab,mask = self.random_flip(img,lab,mask)
            img_batch.append(img)
            lab_batch.append(lab)
            mask_batch.append(mask)
        return img_batch, lab_batch, mask_batch

    def random_crop(self,inp):
        img, lab, mask = inp
        h = random.randint(0,40)
        w = random.randint(0,40)
        img = img[h:h+460,w:w+460]
        lab = lab[h:h+460,w:w+460]
        mask = mask[h:h+460,w:w+460]
        return img,lab,mask

    def random_flip(self,img,lab,mask):
        if random.random()<0.5:
            return img[:,::-1], lab[:,::-1], mask[:,::-1]
        return img, lab, mask

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

reader = data_provider('train_list.txt')
with tf.variable_scope('seg_part'):
	net = network(21)

MAX_ITER = 100000
BSIZE = 1
for i in range(MAX_ITER):
    img_batch, lab_batch, mask_batch = reader.next_batch(BSIZE)
    loss = net.train(img_batch,lab_batch,mask_batch)
    print('Iter:%d\tLoss:%.4f'%(i,loss))

    # monitor the training process
    if i%10==0:
        fname = random.sample(reader.fnames,1)
        img = cv2.imread(fname[0])
        img = img_reader.pad(img,np.uint8)
        img_inp = img_reader.read_img(fname[0],500,True)

        mask = read_label(fname[0].replace('reform_img', 'reform_annot').replace('jpg', 'png'))
        mask = img_reader.pad(mask, np.uint8, False)
        mask[mask != 0] = 1

        img = img[20:480,20:480]
        img_inp = img_inp[20:480,20:480]
        mask = mask[20:480,20:480]

        res = net.eval([img_inp],[mask])
        res = np.uint8(res)
        res = cv2.resize(res,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        cmap = color_map()
        res = cmap[res]
        cv2.imwrite('./sample_seg/%d.jpg'%(i),img)
        cv2.imwrite('./sample_seg/%d_2.jpg'%(i),res)

    # save the network
    if i%1000==0 and i>0:
        net.save('./savings_seg/%d.ckpt'%(i))
