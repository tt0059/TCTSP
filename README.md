# Image Paragraph Captioning with Topic Clustering and Topic Shift Prediction

## Environment settings
The codebase is tested under the following environment settings:
- cuda: 10.1
- python: 3.6.13
- pytorch: 1.4.0
- torchvision: 0.5.0
- cider (already included as a submodule)
- coco-caption (already included as a submodule)

For more detailed environment settings, please refer to TCTSP/environment.yml

## Prepare the data
We have extracted the features of the images in the Stanford image paragraph dataset using Faster R-CNN and uploaded them. 
The way to get them is as follows:

1.download res101_10_100_ray.tar.gz from: https://pan.baidu.com/s/1JrSwDxFDPZLWhaWlGyN12A?pwd=0059.

2.Extract to the TCTSP/ directory using the following command：
```shell
tar -xzvf res101_10_100_ray.tar.gz
```

The rest of the data needed for the experiment is stored in data_vg.tar.gz and uploaded, and the method to obtain is as follows:

1.download data_vg.tar.gz from: https://pan.baidu.com/s/1JrSwDxFDPZLWhaWlGyN12A?pwd=0059.

2.Extract to the TCTSP/ directory using the following command：
```shell
tar -xzvf data_vg.tar.gz
```

## Download the checkpoints

Our already pre-trained model is obtained in the following way:

1.download caption_model_57.pth from: https://pan.baidu.com/s/1JrSwDxFDPZLWhaWlGyN12A?pwd=0059.

2.Put caption_model_57.pth under path ./experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test/snapshot/

## Evaluate
To conduct evaluation of the pre-trained model, you can run the following commands:
```shell
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder ./experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test --resume 57 --markov_mat_path ./data/markov_mat_kmeans.npy
```

## Acknowledgement

Part of the code is borrowed from [image-captioning](https://github.com/JDAI-CV/image-captioning). We thank the authors for releasing their codes.
