### Nested Adversarial Networks (NAN)


- Sample training code of NAN in PASCAL VOC2012. For fine-tuning on MHP v2.0, MHP v1.0, PASCAL-Person-Part and Buffy, please follow the similar procedures as the pre-training on PASCAL VOC2012 and relevant details in our [MHP v2.0 paper](https://arxiv.org/pdf/1804.03287.pdf).

- We are re-working the codes for NAN model [here](https://github.com/ZhaoJ9014/Multi-Human-Parsing/tree/master/Nested_Adversarial_Networks/NAN_rework)

- "train.list" is the training image list: Left column: path of RGB images; Right column: path of annotations.


- Note: the instance segmentation image should be stored in the "SegmentationObject" folder and the object annotation should be stored in the "Annotations" folder, same as the official file organization in PASCAL VOC2012. It is suggested to use SBD for expansion of the training data.


- [Download](https://drive.google.com/drive/folders/1zycuNwILRBNy25ptQeI_DA5yripjhNiD?usp=sharing) and unzip the ImageNet pre-trained WS-ResNet model folder into the root directory.


- Run deploy_pretrain.py. 

- Run train_step1.py, train_step2.py and train_step3.py to perform step-wise training. Run train_e2e to perform end-to-end fine-tuning. The training process will be output into corresponding folders created automatically after you run the scripts.


- Run the sample code output_sample.py to get the output images.


### Citation
- Please consult and consider citing the following paper:


      @article{zhao2018understanding,
      title={Understanding Humans in Crowded Scenes: Deep Nested Adversarial Learning and A New Benchmark for Multi-Human Parsing},
      author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Zhou, Li and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
      journal={arXiv preprint arXiv:1804.03287},
      year={2018}
      }
