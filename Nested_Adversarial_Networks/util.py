import os.path as osp
from PIL import Image

import numpy as np 
import scipy.linalg as la
from numpy.core.umath_tests import inner1d
import pandas as pd 
from matplotlib import pyplot as plt 
from more_itertools import unique_everseen
import numba 
from numba import *

def affinity(s,non0_coord,var):
    n = s.shape[0]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                res = 0
            else:
                p = s[i] - s[j]
                p2 = non0_coord[i] - non0_coord[j]
                res = np.exp(-np.sqrt(np.sum(p*p)) / (2*var)) + np.exp(-np.sqrt(np.sum(p2*p2)) / (2*var))
            A[i,j] = res
    return A

def kmeans(y, k, max_iter=20):
    idx = np.random.choice(len(y), k, replace=False)
    idx_data = y[idx]
    for i in range(max_iter):
        dist = np.array([inner1d(y-c, y-c) for c in idx_data])
        clusters = np.argmin(dist, axis=0)
        idx_data = np.array([y[clusters==i].mean(axis=0) for i in range(k)])
    return clusters, idx_data

def finalclustering(s,non0_coord,k,var):
    n = s.shape[0]
    A = affinity(s,non0_coord,var)
    D = np.zeros((n,n))
    # obtain D^-1
    for i in range(n):
        D[i,i] = 1/(A[i].sum())
    print(D[0])
        
    L = np.sqrt(D).dot(A).dot(np.sqrt(D))
    value, vector = la.eig(L)
    idx = np.argsort(value)[::-1]
    value = value[idx]
    vector = vector[:,idx]
    
    X = vector[:,:k]
    Y = X / np.sum(X,1)[:, np.newaxis]
    
    clusters, data = kmeans(Y,k,max_iter=20)
    final = np.concatenate((s,clusters.reshape((len(clusters),1))),axis=1)
    return final, data, clusters

cfg = {}


def as_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

def get_interp_method(imh_src, imw_src, imh_dst, imw_dst, default=Image.CUBIC):
    if not cfg.get('choose_interpolation_method', False):
        return default
    if imh_dst < imh_src and imw_dst < imw_src:
        return Image.ANTIALIAS
    elif imh_dst > imh_src and imw_dst > imw_src:
        return Image.CUBIC
    else:
        return Image.LINEAR

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

def get_output_img(msk,seg,inst_num,coord):
    imgshape = msk.shape
    msk_img = np.argmax(msk,-1)
    seg_img = np.argmax(seg,-1)
    inst_num = np.rint(inst_num)

    instance_img = np.zeros(imgshape,dtype=np.uint8)
    non0 = np.nonzero(msk_img)
    coord_map = coord[non0[0],non0[1]]
    coord_map = coord_map[:2]
    indices = np.transpose(non0)
    final, data, clusters = finalclustering(coord_map, indices, inst_num, 0.5)
    instance_img[final[0],final[1]] = final[2]
    cmap = color_map()

    msk_img[msk_img==1] = 255
    msk_img = np.uint8(msk_img)
    seg_img = cmap[seg_img]
    instance_img = cmap[instance_img]

    return msk_img, seg_img, instance_img

