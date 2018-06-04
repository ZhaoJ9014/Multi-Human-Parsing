### Nested Adversarial Networks (NAN)


- Sample training code of NAN in PASCAL VOC2012. For fine-tuning on MHP v2.0, MHP v1.0, PASCAL-Person-Part and Buffy, please follow the similar procedures as the pre-training on PASCAL VOC2012 and relevant details in our paper.


- "train.list" is the training image list: Left column: path of RGB images; Right column: path of annotations.


- Note: the instance segmentation image should be stored in the "SegmentationObject" folder and the object annotation should be stored in the "Annotations" folder, same as the official file organization in PASCAL VOC2012. It is suggested to use SBD for expansion of the training data.


- Download and unzip the ImageNet pre-trained WS-ResNet "model" folder into the root directory.


- Run train_step1.py, train_step2.py and train_step3.py to perform the step-wise training (the end-to-end version will come soon). The training process will be output into corresponding folders created automatically after you run the scripts.
