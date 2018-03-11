## Multi-Human-Parsing (MHP)
Jianshu Li, Jian Zhao, Yunchao Wei, Congyan Lang, Yidong Li, Jiashi Feng, "Towards Real World Human Parsing: Multiple-Human Parsing in the Wild", ArXiv 2017.


### Introduction
The recent progress of human parsing techniques has been largely driven by the availability of rich data resources. However, there still exist some critical discrepancies between the current benchmark datasets and the real world human parsing scenarios. For instance, all the human parsing datasets only contain one person per image, while usually multiple persons appear simultaneously in a realistic scene. It is more practically demanded to simultaneously parse multiple persons, which presents a greater challenge to modern human parsing methods. Unfortunately, absence of relevant data resources severely impedes the development of multiple-human parsing methods. To facilitate future human parsing research, we introduce the first Multiple-Human Parsing (MHP) benchmark dataset, which contains multiple persons in a real world scene per single image. The MHP dataset contains various numbers of persons (from 2 to 16) per image with 18 semantic classes for each parsing annotation. Persons appearing in the MHP images present sufficient variations in pose, occlusion and interaction. To tackle the multiple-human parsing problem, we also propose a novel Multiple-Human Parser (MH-Parser) as a reference method, which considers both the global context and local cues for each person in the parsing process. The model is demonstrated to outperform the naive "detect-and-parse" approach by a large margin, which will serve as a solid baseline and help drive the future research in real world human parsing.


To our best knowledge, we are the first to propose a new Multi-Human Parsing task. We construct a novel and comprehensive Multi-Human Parsing (MHP) benchmark dataset with fine-grained pixel-wise annotations and well designed baselines to push the frontiers of relevant research ([wechat news](https://mp.weixin.qq.com/s/tfiPHvkhPW4HDEUzBMseGQ)). Please consult and consider citing the following papers:


   @article{li2017towards,
   title={Towards Real World Human Parsing: Multiple-Human Parsing in the Wild},
   author={Li, Jianshu and Zhao, Jian and Wei, Yunchao and Lang, Congyan and Li, Yidong and Feng, Jiashi},
   journal={arXiv preprint arXiv:1705.07206},
   year={2017}
   }
  
  
  ### Multi-Human Parsing (MHP) Dataset
  The MHP v1.0 dataset is also available at [google drive](https://drive.google.com/file/d/1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5/view?usp=sharing) and [baidu drive](https://pan.baidu.com/s/1mjTtWqW) (password: cmtp).
