### Nested Adversarial Networks (NAN)


- "train.list" is the your training image list: Left column: path of original image; Right column: path of annotation.


- Note: the instance segmentation image should be stored in the "SegmentationObject" folder and the object annotation should be stored in the "Annotations" folder, as the official file organization in VOC 2012. It is suggested to use SBD for expansion of the training data.


- Download and unzip the "model" folder into the root directory.


- Run train_step1.py, train_step2.py and train_step3.py to perform the step-wise training (the end-to-end version will come soon). The training process will be output into corresponding folders created automatically after you run the scripts.
