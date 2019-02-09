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
if not os.path.exists('./savings_bgfg/'):
    os.mkdir('./savings_bgfg/')
if not os.path.exists('./sample_bgfg/'):
    os.mkdir('./sample_bgfg/')


class network():
    def __init__(self):
        inp_holder = tf.placeholder(tf.float32,[None,460,460,3])
        lab_holder = tf.placeholder(tf.int32,[None,460,460])

        self.net_body = seg_main_body(inp_holder)
        seg_layer = self.segmentation_layer(self.net_body.feature_layer,12)
        self.build_loss(seg_layer,lab_holder)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        M.loadSess('./savings_bgfg/',self.sess,init=True,var_list=M.get_trainable_vars('bg_fg/WideRes'))

        self.inp_holder = inp_holder
        self.lab_holder = lab_holder
        self.seg_layer = seg_layer

    def segmentation_layer(self,feature,dilation):
        with tf.variable_scope('SegLayer'):
            mod = M.Model(feature)
            mod.convLayer(3,2,dilation_rate=dilation)
        return mod.get_current_layer()

    def build_loss(self,seg_layer,lab_holder):
        lab_reform = tf.expand_dims(lab_holder, -1) # 460 x 460 x 1
        seg_layer = tf.image.resize_images(seg_layer, tf.shape(lab_reform)[1:3]) # 460 x 460 x 2
        lab_reform = tf.squeeze(lab_reform, axis=3) # 460 x 460 x 2
        seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_layer,labels=lab_reform))

        var_s = M.get_trainable_vars('bg_fg/SegLayer')
        var_net = M.get_trainable_vars('bg_fg/WideRes')

        train_step = tf.train.AdamOptimizer(0.0001).minimize(seg_loss,var_list=var_s)
        train_step2 = tf.train.AdamOptimizer(1e-6).minimize(seg_loss,var_list=var_net)
        upds = M.get_update_ops()
        with tf.control_dependencies(upds+[train_step,train_step2]):
            train_op = tf.no_op()

        self.loss = seg_loss
        self.train_op = train_op

    def train(self,img_batch,lab_batch):
        ls,_ = self.sess.run([self.loss,self.train_op],feed_dict={self.inp_holder:img_batch, self.lab_holder:lab_batch})
        return ls

    def eval(self,img_batch):
        seg_res = self.sess.run(self.seg_layer,feed_dict={self.inp_holder:img_batch})
        return seg_res

    def save(self,path):
        self.saver.save(self.sess,path)

    def load(self,path):
        M.loadSess(sess=self.sess,modpath=path)

    def eval(self,img_batch):
        res = self.sess.run(self.seg_layer,feed_dict={self.inp_holder:img_batch})
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
            segfile = i
            jpgfile = i.replace('SegmentationClass','JPEGImages').replace('.png','.jpg')
            jpg = img_reader.read_img(jpgfile,500,padding=True)
            seg = self.read_label(segfile)
            seg = img_reader.pad(seg,np.uint8,False)
            seg[seg!=15] = 0
            seg[seg==15] = 1
            if seg.shape[0]!=500:
                print(segfile)
                continue
            self.data.append([jpg,seg])
            self.fnames.append(jpgfile)
            cnt += 1
            if cnt%100==0:
                print(cnt)
        print('Data length:',len(self.data))

    def next_batch(self,BSIZE):
        res = random.sample(self.data,BSIZE)
        img_batch = []
        lab_batch = []
        for i in range(BSIZE):
            img,lab = self.random_crop(res[i])
            img,lab = self.random_flip(img,lab)
            img_batch.append(img)
            lab_batch.append(lab)
        return img_batch, lab_batch

    def random_crop(self,inp):
        img, lab = inp
        h = random.randint(0,40)
        w = random.randint(0,40)
        img = img[h:h+460,w:w+460]
        lab = lab[h:h+460,w:w+460]
        return img,lab

    def random_flip(self,img,lab):
        if random.random()<0.5:
            return img[:,::-1], lab[:,::-1]
        return img, lab

    def read_label(self,path):
        im = Image.open(path)
        img = np.array(im,np.uint8)
        return img

reader = data_provider('train.list')

with tf.variable_scope('bg_fg'):
    net = network()

MAX_ITER = 100000
BSIZE = 1
for i in range(MAX_ITER):
    img_batch, lab_batch = reader.next_batch(BSIZE)
    loss = net.train(img_batch,lab_batch)
    print('Iter:%d\tLoss:%.4f'%(i,loss))

    # monitor the training process
    if i%10==0:
        fname = random.sample(reader.fnames,1)
        img = cv2.imread(fname[0])
        img = img_reader.pad(img,np.uint8)
        img_inp = img_reader.read_img(fname[0],500,True)

        img = img[20:480,20:480]
        img_inp = img_inp[20:480,20:480]

        res = net.eval([img_inp])
        res = np.uint8(res)
        res = cv2.resize(res,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        res[res==1] = 255
        cv2.imwrite('./sample_bgfg/%d.jpg'%(i),img)
        cv2.imwrite('./sample_bgfg/%d_2.jpg'%(i),res)

    # save model
    if i%1000==0 and i>0:
        net.save('./savings_bgfg/%d.ckpt'%(i))
