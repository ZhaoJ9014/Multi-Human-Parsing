import sys, os
import numpy as np
import cv2
import pickle, gzip
from tqdm import tqdm, trange
from voc_eval import voc_ap
import scipy.sparse

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def cal_one_mean_iou(image_array, label_array, NUM_CLASSES):
    hist = fast_hist(label_array, image_array, NUM_CLASSES).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu

def get_gt(list_dat):
    class_recs = {}
    npos = 0

    for dat in tqdm(list_dat, desc='Loading gt..'):
        imagename = dat['filepath'].split('/')[-1]
        if len(dat['bboxes']) == 0:
            gt_box=np.array([])
            det = []
            anno_adds = []
        else:
            gt_box = []
            anno_adds = []
            for bbox in dat['bboxes']:
                mask_gt = cv2.imread(bbox['ann_path'], cv2.IMREAD_GRAYSCALE)
                if np.sum(mask_gt>0)==0: continue
                if np.allclose(mask_gt==255, mask_gt>0): continue # ignore label
                anno_adds.append(bbox['ann_path'])
                gt_box.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                npos = npos + 1 

            det = [False] * len(anno_adds)
        class_recs[imagename] = {'gt_box': np.array(gt_box),
                                 'anno_adds': anno_adds, 
                                 'det': det}
    return class_recs, npos

    
def eval_seg_ap(results_all, dat_list, nb_class=19, ovthresh_seg=0.5, Sparse=False, From_pkl=False):
    '''
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    '''
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:4])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]


    class_recs, npos = get_gt(dat_list)
    nd = len(image_ids)
    tp_seg = np.zeros(nd)
    fp_seg = np.zeros(nd)
    pcp_list= []

    for d in trange(nd, desc='Finding AP^P at thres %f..'%ovthresh_seg):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        jmax = -1
        if From_pkl:
            results = pickle.load(gzip.open(results_all[image_ids[d]]))
        else:
            results = results_all[image_ids[d]]

        mask0 = results['MASKS'][Local_segs_ptr[d]]
        if Sparse:
            mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
        else:
            mask_pred = mask0.astype(np.int)

        for i in xrange(len(R['anno_adds'])):
            mask_gt = cv2.imread(R['anno_adds'][i], cv2.IMREAD_GRAYSCALE)
            seg_iou= cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

            mean_seg_iou = np.nanmean(seg_iou)
            if mean_seg_iou > ovmax:
                ovmax =  mean_seg_iou
                seg_iou_max = seg_iou 
                jmax = i
                mask_gt_u = np.unique(mask_gt)

        if ovmax > ovthresh_seg:
            if not R['det'][jmax]:
                tp_seg[d] = 1.
                R['det'][jmax] = 1
                pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg))
                if pcp_d > 0:
                    pcp_list.append(pcp_n/pcp_d)
                else:
                    pcp_list.append(0.0)
            else:
                fp_seg[d] =  1.
        else:
            fp_seg[d] =  1.

    # compute precision recall
    fp_seg = np.cumsum(fp_seg)
    tp_seg = np.cumsum(tp_seg)
    rec_seg = tp_seg / float(npos)
    prec_seg = tp_seg / (tp_seg + fp_seg)

    ap_seg = voc_ap(rec_seg, prec_seg)

    assert(np.max(tp_seg) == len(pcp_list)), "%d vs %d"%(np.max(tp_seg),len(pcp_list))
    pcp_list.extend([0.0]*(npos - len(pcp_list)))
    pcp = np.mean(pcp_list)

    print 'AP_seg, PCP:', ap_seg, pcp
    return ap_seg, pcp


def get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, cache_pkl_path='tmp/', Sparse=False):
    '''
    cache_pkl: if the memory can't hold all the results, set cache_pkl to be true to pickle down the results 
    Sparse: Sparsify the masks to save memory
    '''
    results_all = {}
    for dat in tqdm(dat_list, desc='Generating predictions ..'):
        results = {} 

        dets, masks = [], []
        for box in dat['bboxes']:
            mask_gt = cv2.imread(box['ann_path'], cv2.IMREAD_GRAYSCALE)
            if np.sum(mask_gt)==0: continue
            ys, xs = np.where(mask_gt>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            dets.append((x1, y1, x2, y2, 1.0))
            masks.append(mask_gt)            

        # Reuiqred Field of each result: a list of masks, each is a multi-class masks for one person.
            # It can also be sparsified to [scipy.sparse.csr_matrix(mask) for mask in masks] to save memory cost
        results['MASKS']= masks if not Sparse else [scipy.sparse.csr_matrix(mask) for mask in masks]
        # Reuiqred Field of each result, a list of detections corresponding to results['MASKS']. 
        results['DETS'] = dets    

        key = dat['filepath'].split('/')[-1]
        if cache_pkl:
            results_cache_add = cache_pkl_path + key + '.pklz'
            pickle.dump(results, gzip.open(results_cache_add, 'w'))
            results_all[key] = results_cache_add
        else:
            results_all[key]=results
    return results_all


def test_mhp():
    import mhp_data
    dat_list  = mhp_data.get_val_dat()
    NUM_CLASSES = 19
    results_all = get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, Sparse=True)
    eval_seg_ap(results_all, dat_list, From_pkl=False, Sparse=True)
        

if __name__ == '__main__':
    test_mhp()
