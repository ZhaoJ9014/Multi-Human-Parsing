### Nested Adversarial Networks (NAN)



'train.list' is the sample of training image list. 
Left column: path of original image
Right column: path of segmentation image

Note: the instance segmentation image should be stored in the 'SegmentationObject' folder and the object annotation should be stored in the 'Annotations' folder. (as the official file organization in VOC2012)
*It is suggested to use SBD for expansion of the training data.

Download and unzip the 'model' folder into the root directory.

Just run train_step1, train_step2 and train_step3 will perform the step-wise training. 
The training process will be output into corresponding folders created automatically after you run the scripts.
