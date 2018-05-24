## Multi-Human Parsing Version 2.0 Evaluation
- mhp_data.py: This script generates a list of data, and also visualizes the dataset.


- eval_mhp.py: This script evaluates the predictions. It generates a set of perfect predictions with the ground truth, and evluates the perfect predictions. To evaluate your algorithm, replace the results ['MASKS'] and results ['DETS'] with the output of your algorithm.


- eval_sumission.py: This script takes in the format of submission (https://lv-mhp.github.io/evaluate) and outputs the metrics. 


- voc_eval.py: A helper script.


- Note: Due to the data saving format of our annotation tool, all ground truth files have 3 channels. Please perform simple data pre-processing by only utilizing 1 channel for training and testing purposes. To load the ground truths, please use mask_gt = np.array(Image.open(R['anno_adds'][i])),if len(mask_gt.shape)==3:mask_gt=mask_gt[:,:,0].
