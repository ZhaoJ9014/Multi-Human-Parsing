### Multi-Human-Parsing (MHP)


### Originality
- To our best knowledge, we are the first to propose a new Multi-Human Parsing task, corresponding datasets and baseline methods.


### Task Definition
- Multi-Human Parsing refers to partitioning a crowd scene image into semantically consistent regions belonging to the body parts or clothes items while differentiating different identities, such that each pixel in the image is assigned a semantic part label, as well as the identity it belongs to. A lot of higher-level applications can be founded upon Multi-Human Parsing, such as virtual reality, automatic production recommendation, video surveillance, and group behavior analysis.


### Motivation
- The Multi-Human Parsing project of Learning and Vision (LV) Group, National University of Singapore (NUS) is proposed to push the frontiers of fine-grained visual understanding of humans in crowd scene. 


- Multi-Human Parsing is significantly different from traditional well-defined object recognition tasks, such as object detection, which only provides coarse-level predictions of object locations (bounding boxes); instance segmentation, which only predicts the instance-level mask without any detailed information on body parts and fashion categories; human parsing, which operates on category-level pixel-wise prediction without differentiating different identities. 


- In real world scenario, the setting of multiple persons with interactions are more realistic and usual. Thus a task, corresponding datasets and baseline methods to consider both the fine-grained semantic information of each individual person and the relationships and interactions of the whole group of people are highly desired.


### Multi-Human Parsing (MHP) v1.0 Dataset
<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig1.png" width="1000px"/>


<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig2.png" width="1000px"/>


- Statistics: The MHP v1.0 dataset contains 4,980 images, each with at least two persons (average is 3). We randomly choose 980 images and their corresponding annotations as the testing set. The rest form a training set of 3,000 images and a validation set of 1,000 images. For each instance, 18 semantic categories are defined and annotated except for the "background" category, i.e. “hat”, “hair”, “sunglasses”, “upper clothes”, “skirt”, “pants”, “dress”, “belt”, “left shoe”, “right shoe”, “face”, “left leg”, “right leg”, “left arm”, “right arm”, “bag”, “scarf” and “torso skin”. Each instance has a complete set of annotations whenever the corresponding category appears in the current image. 


- Download: The MHP v1.0 dataset is available at [google drive](https://drive.google.com/file/d/1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5/view?usp=sharing) and [baidu drive](https://pan.baidu.com/s/1mjTtWqW) (password: cmtp).


### CVPR VUHCS2018 Workshop
- We will organize the CVPR 2018 Workshop on Visual Understanding of Humans in Crowd Scene ([VUHCS2018](http://cvpr2018.thecvf.com/program/workshops)). This workshop is collaborated by NUS, CMU, and SYSU. Based on VUHCS2017, we have further strengthened this workshop by augmenting it with 6 competition tracks: the single-person human parsing, the single-person pose estimation, the multi-person human parsing, the multi-person video parsing, the fine-grained multi-human parsing, and the multi-human pose estimation, which are expected to be released on 04/2018.


- [Call for Papers](https://www.ece.nus.edu.sg/lv/files/VUHCS2018.pdf)


### Citation
- [wechat news](https://mp.weixin.qq.com/s/tfiPHvkhPW4HDEUzBMseGQ). Please consult and consider citing the following papers:


      @article{li2017towards,
      title={Towards Real World Human Parsing: Multiple-Human Parsing in the Wild},
      author={Li, Jianshu and Zhao, Jian and Wei, Yunchao and Lang, Congyan and Li, Yidong and Feng, Jiashi},
      journal={arXiv preprint arXiv:1705.07206},
      year={2017}
      }
