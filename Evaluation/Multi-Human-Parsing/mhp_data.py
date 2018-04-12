import os
import cv2
import numpy as np
import pickle
from tqdm import trange, tqdm

data_root = '/pathe/to/LV-MHP-v1/'
img_root = data_root + 'images/'
ann_root = data_root + 'annotations/'
global_seg_root = data_root + 'global_seg/'


def obtain_ann_dict(img_root, ann_root):
    ann_dict={}
    assert(os.path.isdir(img_root)), 'Path does not exist: {}'.format(img_root)
    assert(os.path.isdir(ann_root)), 'Path does not exist: {}'.format(ann_root)

    for add in os.listdir(img_root):
        ann_dict[add]=[]

    for add in os.listdir(ann_root):
        ann_dict[add[0:4]+'.jpg'].append(add)
    return ann_dict

ann_dict =  obtain_ann_dict(img_root, ann_root)

def get_train_dat():
    src = 'train_list.txt'
    return get_data(src)

def get_val_dat():
    src = 'val_list.txt'
    return get_data(src)

def get_test_dat():
    src = 'test_list.txt'
    return get_data(src)


def get_data(src):
    flist=[line.strip()  for line in open(data_root+src).readlines()]
    list_dat=[]
    for add in tqdm(flist, desc='Loading %s ..' %src):
        dat={}
        im_sz=cv2.imread(img_root + add).shape
        dat['filepath'] = img_root + add
        dat['global_mask_add'] = global_seg_root + add.replace('.jpg', '.png')

        dat['width'] = im_sz[1]
        dat['height'] = im_sz[0]
        dat['bboxes'] = []
        for ann_add in sorted(ann_dict[add]):
            ann=cv2.imread(ann_root+ann_add, cv2.IMREAD_GRAYSCALE)
            ys, xs=np.where(ann>0)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            dat['bboxes'].append(
                {'class': 'person',
                'ann_path': ann_root+ann_add, 
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2 } )

        list_dat.append(dat)

    return list_dat


def show_data(list_dat, num=4):    
    from pylab import plt
    for dat in np.random.choice(list_dat, num):
        print dat
        im=cv2.imread(dat['filepath'])[:,:,::-1]
        plt.figure(1)
        plt.imshow(im)
        for bbox in dat['bboxes']:
            plt.gca().add_patch(plt.Rectangle((bbox['x1'], bbox['y1']),
                      bbox['x2'] - bbox['x1'],
                      bbox['y2'] - bbox['y1'], fill=False,
                      edgecolor='red', linewidth=1) )
        for idx, bbox in enumerate(dat['bboxes']):
            ann=cv2.imread(bbox['ann_path'], cv2.IMREAD_GRAYSCALE)
            plt.figure(11+idx)
            plt.imshow(ann)
        plt.show()

if __name__ == '__main__':
    dat_list = get_val_dat()
    show_data(dat_list)

    dat_list = get_train_dat()
    show_data(dat_list)

    dat_list = get_test_dat()
    show_data(dat_list)
