### Multi-Human-Parsing (MHP)

:star: ACM MM'18 [Best Student Paper](http://www.acmmm.org/2018/awards/)

### Originality
- To our best knowledge, we are the first to propose a new Multi-Human Parsing task, corresponding datasets, evaluation metrics and baseline methods.


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


- [WeChat News](https://mp.weixin.qq.com/s/tfiPHvkhPW4HDEUzBMseGQ). 


- Download: The MHP v1.0 dataset is available at [google drive](https://drive.google.com/file/d/1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5/view?usp=sharing) and [baidu drive](https://pan.baidu.com/s/1mjTtWqW) (password: cmtp).

- Please refer to our [MHP v1.0 paper](https://arxiv.org/pdf/1705.07206.pdf) (submitted to IJCV) for more details.


### Multi-Human Parsing (MHP) v2.0 Dataset
<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig3.png" width="1000px"/>


<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig4.png" width="1000px"/>


<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig5.png" width="1000px"/>


- Statistics: The MHP v2.0 dataset contains 25,403 images, each with at least two persons (average is 3). We randomly choose 5,000 images and their corresponding annotations as the testing set. The rest form a training set of 15,403 images and a validation set of 5,000 images. For each instance, 58 semantic categories are defined and annotated except for the "background" category, i.e. "cap/hat", "helmet", "face", "hair", "left-arm", "right-arm", "left-hand", "right-hand", "protector", "bikini/bra", "jacket/windbreaker/hoodie", "t-shirt", "polo-shirt", "sweater", "singlet", "torso-skin", "pants", "shorts/swim-shorts", "skirt", "stockings", "socks", "left-boot", "right-boot", "left-shoe", "right-shoe", "left-highheel", "right-highheel", "left-sandal", "right-sandal", "left-leg", "right-leg", "left-foot", "right-foot", "coat", "dress", "robe", "jumpsuit", "other-full-body-clothes", "headwear", "backpack", "ball", "bats", "belt", "bottle", "carrybag", "cases", "sunglasses", "eyewear", "glove", "scarf", "umbrella", "wallet/purse", "watch", "wristband", "tie", "other-accessary", "other-upper-body-clothes" and "other-lower-body-clothes". Each instance has a complete set of annotations whenever the corresponding category appears in the current image. Moreover, 2D human poses with 16 dense key points ("right-shoulder", "right-elbow", "right-wrist", "left-shoulder", "left-elbow", "left-wrist", "right-hip", "right-knee", "right-ankle", "left-hip", "left-knee", "left-ankle", "head", "neck", "spine" and "pelvis". Each key point has a flag indicating whether it is visible-0/occluded-1/out-of-image-2) and head & instance bounding boxes are also provided to facilitate Multi-Human Pose Estimation research. 


- Download: The MHP v2.0 dataset is available at [google drive](https://drive.google.com/file/d/1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn/view?usp=sharing) and [baidu drive](https://pan.baidu.com/s/1BvdoyKgm-RlINBGghmvrOQ) (password: uxrb).


- Please refer to our [MHP v2.0 paper](https://arxiv.org/pdf/1804.03287.pdf) (ACM MM'18 [Best Student Paper](http://www.acmmm.org/2018/awards/)) for more details.


### Evaluation Metrics
- Multi-Human Parsing: We use two human-centric metrics for multi-human parsing evaluation, which are initially reported by our [MHP v1.0 paper](https://arxiv.org/pdf/1705.07206.pdf). The two metrics are Average Precision based on part (AP<sup>p</sup>) (%) and Percentage of Correctly parsed semantic Parts (PCP) (%). For evaluation code, please refer to the "Evaluation" folder under our "Multi-Human-Parsing_MHP" repository.


- Multi-Human Pose Estimation: Followed MPII, we use mAP (%) evaluation measure.


### CVPR VUHCS2018 Workshop
- We have organized the CVPR 2018 Workshop on Visual Understanding of Humans in Crowd Scene ([VUHCS 2018](https://vuhcs.github.io/#portfolio)). This workshop is collaborated by NUS, CMU and SYSU. Based on VUHCS 2017, we have further strengthened this Workshop by augmenting it with 5 competition tracks: the single-person human parsing, the multi-person human parsing, the single-person pose estimation, the multi-human pose estimation and the fine-grained multi-human parsing.


- [Result Submission & Leaderboard](https://lv-mhp.github.io/).


- [WeChat News](https://mp.weixin.qq.com/s/0W3AOMTeOyngnM1WA-8C5w).



****
### Donation 
:moneybag:

* Your donation is highly welcomed to help us further develop the Multi-Human Parsing project to better facilitate more cutting-edge researches and applications on human-centric multi-media understanding. The donation QR code via Wechat is as below. Appreciate it very much:heart:
 
  <img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing/blob/master/Figures/Donation.jpeg" width="200px"/>
  

### Citation
- Please consult and consider citing the following papers:


      @article{zhao2018understanding,
      title={Understanding Humans in Crowded Scenes: Deep Nested Adversarial Learning and A New Benchmark for Multi-Human Parsing},
      author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Zhou, Li and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
      journal={arXiv preprint arXiv:1804.03287},
      year={2018}
      }


      @article{li2017towards,
      title={Multi-Human Parsing in the Wild},
      author={Li, Jianshu and Zhao, Jian and Wei, Yunchao and Lang, Congyan and Li, Yidong and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
      journal={arXiv preprint arXiv:1705.07206},
      year={2017}
      }
