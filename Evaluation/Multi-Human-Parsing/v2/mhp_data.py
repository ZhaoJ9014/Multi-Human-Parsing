import os
import cv2
import numpy as np
import pickle
from tqdm import trange, tqdm
from PIL import Image


def obtain_ann_dict(img_root, ann_root):
    ann_dict = {}
    assert (os.path.isdir(img_root)), 'Path does not exist: {}'.format(img_root)
    assert (os.path.isdir(ann_root)), 'Path does not exist: {}'.format(ann_root)

    for add in os.listdir(img_root):
        ann_dict[add] = []

    for add in os.listdir(ann_root):
        ann_dict[add[:-10] + '.jpg'].append(add)
    return ann_dict


def get_data(data_root, set_):
    assert (set_ in ['train', 'val', 'test_all', 'test_inter_top20', 'test_inter_top10'])

    set_list_add = set_ + '.txt'
    if set_.startswith('test'):
        set_ = set_.split('_')[0]

    list_root = data_root + '/list/'
    img_root = data_root + set_ + '/images/'
    ann_root = data_root + set_ + '/parsing_annos/'  # '/annotations/'

    ann_dict = obtain_ann_dict(img_root, ann_root)

    flist = [line.strip() for line in open(list_root + set_list_add).readlines()]

    list_dat = []
    for add in tqdm(flist, desc='Loading %s ..' % (set_list_add)):
        dat = {}
        im_sz = cv2.imread(img_root + add + '.jpg').shape
        dat['filepath'] = img_root + add + '.jpg'

        dat['width'] = im_sz[1]
        dat['height'] = im_sz[0]
        dat['bboxes'] = []
        for ann_add in sorted(ann_dict[add + '.jpg']):
            ann = np.array(Image.open(ann_root + ann_add))
            if len(ann.shape) == 3:
                ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
            ys, xs = np.where(ann > 0)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            dat['bboxes'].append(
                {'class': 'person',
                 'ann_path': ann_root + ann_add,
                 'x1': x1,
                 'y1': y1,
                 'x2': x2,
                 'y2': y2})

        list_dat.append(dat)

    return list_dat


def show_data(list_dat, num=4):
    from pylab import plt
    for dat in np.random.choice(list_dat, num):
        print(dat)
        im = cv2.imread(dat['filepath'])[:, :, ::-1]
        plt.figure(1)
        plt.imshow(im)
        for bbox in dat['bboxes']:
            plt.gca().add_patch(plt.Rectangle((bbox['x1'], bbox['y1']),
                                              bbox['x2'] - bbox['x1'],
                                              bbox['y2'] - bbox['y1'], fill=False,
                                              edgecolor='red', linewidth=1))
        for idx, bbox in enumerate(dat['bboxes']):
            ann = np.array(Image.open(bbox['ann_path']))
            if len(ann.shape) == 3:
                ann = ann[:, :, 0]  # Make sure ann is a two dimensional np array.
            plt.figure(11 + idx)
            plt.imshow(ann)
        plt.show()


def cache_dat_list():
    for set_ in ['val', 'test_all', 'test_inter_top20', 'test_inter_top10']:
        print('Caching {}..'.format(set_))
        dat_list_train = get_data(data_root, set_)
        pickle.dump(dat_list_train, open('cache/dat_list_{}.pkl'.format(set_), 'w'))


if __name__ == '__main__':
    data_root = '/home/lijianshu/MultiPerson/data/LV-MHP-v2/'
    # Possible values for set_: 'train', 'val', 'test_all', 'test_inter_top20', 'test_inter_top10'
    set_ = 'train'
    dat_list = get_data(data_root, set_)
    show_data(dat_list)
