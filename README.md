# Image Paragraph Captioning with Topic Clustering and Topic Shift Prediction

This paper has been published in the journal "Knowledge-Based Systems".

## Paper Details
- Title: Image Paragraph Captioning with Topic Clustering and Topic Shift Prediction
- Authors: Ting Tang, Jiansheng Chen, Yiqing Huang, Huimin Ma, Yudong Zhang, Hongwei Yu
- Journal: Knowledge-Based Systems

## Access and Download
The paper can be accessed and downloaded via the following link:
[Download Paper](https://authors.elsevier.com/a/1iTtG3OAb9Cy9i)

## Abstract
Image paragraph captioning involves generating a semantically coherent paragraph describing an image’s visual content. The selection and shifting of sentence topics are critical when a human describes an image. However, previous hierarchical image paragraph captioning methods have not fully explored or utilized sentence topics. In particular, the continuous and implicit modeling of topics in these methods makes it difficult to supervise the topic prediction process explicitly. We propose a new method called topic clustering and topic shift prediction (TCTSP) to solve this problem. Topic clustering (TC) in the sentence embedding space generates semantically explicit and discrete topic labels that can be directly used to supervise topic prediction. By introducing a topic shift probability matrix that characterizes human topic shift patterns, topic shift prediction (TSP) predicts subsequent topics that are both logical and consistent with human habits based on visual features and language context. TCTSP can be combined with various image paragraph captioning model structures to improve performance. Extensive experiments were conducted on the Stanford image paragraph dataset, and superior results were reported compared with previous state-of-the-art approaches. In particular, TCTSP improved the consensus-based image description evaluation (CIDEr) performance of image paragraph captioning to 41.67%. The codes are available at https://github.com/tt0059/TCTSP.

## Citation
For citing this paper, please use the following format:
@article{TANG2024111401,
title = {Image paragraph captioning with topic clustering and topic shift prediction},
journal = {Knowledge-Based Systems},
volume = {286},
pages = {111401},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.111401},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124000364},
author = {Ting Tang and Jiansheng Chen and Yiqing Huang and Huimin Ma and Yudong Zhang and Hongwei Yu},
keywords = {Image paragraph captioning, Topic clustering, Topic shift prediction, Hierarchical supervision},
abstract = {Image paragraph captioning involves generating a semantically coherent paragraph describing an image’s visual content. The selection and shifting of sentence topics are critical when a human describes an image. However, previous hierarchical image paragraph captioning methods have not fully explored or utilized sentence topics. In particular, the continuous and implicit modeling of topics in these methods makes it difficult to supervise the topic prediction process explicitly. We propose a new method called topic clustering and topic shift prediction (TCTSP) to solve this problem. Topic clustering (TC) in the sentence embedding space generates semantically explicit and discrete topic labels that can be directly used to supervise topic prediction. By introducing a topic shift probability matrix that characterizes human topic shift patterns, topic shift prediction (TSP) predicts subsequent topics that are both logical and consistent with human habits based on visual features and language context. TCTSP can be combined with various image paragraph captioning model structures to improve performance. Extensive experiments were conducted on the Stanford image paragraph dataset, and superior results were reported compared with previous state-of-the-art approaches. In particular, TCTSP improved the consensus-based image description evaluation (CIDEr) performance of image paragraph captioning to 41.67%. The codes are available at https://github.com/tt0059/TCTSP.}
}

## Environment settings
The codebase is tested under the following environment settings:
- cuda: 10.1
- numpy 1.19.5
- python: 3.6.13
- pytorch: 1.4.0
- torchvision: 0.5.0
- [coco-caption](https://github.com/ruotianluo/coco-caption) (put pycocoevalcap under path TCTSP/)

For more detailed environment settings, please refer to TCTSP/environment.yml:
```shell
conda env create -f environment.yml
```

## Prepare the data
### Visual feature
We have extracted the features of the images in the Stanford image paragraph dataset using Faster R-CNN and uploaded them. 
The way to get them is as follows:

1. Download res101_10_100_ray.tar.gz from: [https://drive.google.com/file/d/1-17LEg4CEHW2rICjJ_YEfJkpZ8X2PiuZ/view?usp=sharing](https://drive.google.com/file/d/1-17LEg4CEHW2rICjJ_YEfJkpZ8X2PiuZ/view?usp=sharing).

2. Extract to the TCTSP/ directory using the following command：
```shell
tar -xzvf res101_10_100_ray.tar.gz
```

### Others
The rest of the data needed for the experiment is stored in data_vg.tar.gz and uploaded, and the method to obtain is as follows:

1. Download data_vg.tar.gz from: [https://drive.google.com/file/d/1--thaTlTnc6BWU16rV3xa6UEUa5zR6y5/view?usp=sharing](https://drive.google.com/file/d/1--thaTlTnc6BWU16rV3xa6UEUa5zR6y5/view?usp=sharing).

2. Extract to the TCTSP/ directory using the following command：
```shell
tar -xzvf data_vg.tar.gz
```

## Download the checkpoint

Our pre-trained model is obtained in the following way:

1. Download caption_model_57.pth from: [https://drive.google.com/file/d/1-1M8ySZd0FsDMYvdXoa_T8rRsDDK5MLC/view?usp=sharing](https://drive.google.com/file/d/1-1M8ySZd0FsDMYvdXoa_T8rRsDDK5MLC/view?usp=sharing).

2. Make a snapshot folder:
```shell
mkdir ./experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test/snapshot/
```

3. Put caption_model_57.pth under path TCTSP/experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test/snapshot/

## Evaluate
In image paragraph captioning task, we only compute BLEU, METEOR and CIDEr, so other metrics in line 47 of TCTSP/pycocoevalcap/eval.py need to be delete.

To conduct evaluation of the pre-trained model, you can run the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder ./experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test --resume 57 --markov_mat_path ./data/markov_mat_kmeans.npy
```

## Acknowledgement

Part of the code is borrowed from [image-captioning](https://github.com/JDAI-CV/image-captioning). We thank the authors for releasing their codes.
