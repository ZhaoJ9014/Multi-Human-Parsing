## Multi-Human Parsing Version 1.0 Evaluation
- mhp_data.py: This script generates a list of data, and also visualizes the dataset.
  
  
- eval_mhp.py: This script evaluates the predictions. It generates a set of perfect predictions with the ground truth, and evluates the perfect predictions. To evaluate your algorithm, replace the results ['MASKS'] and results ['DETS'] with the output of your algorithm.

- Note: to load the ground truths, please use mask_gt = cv2.imread(R['anno_adds'][i],cv2.IMREAD_GRAYSCALE).
