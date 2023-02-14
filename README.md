# Image Paragraph Captioning with Topic Clustering and Topic Shift Prediction

## Environment settings
The codebase is tested under the following environment settings:
- cuda: 10.1
- python: 3.6.13
- pytorch: 1.4.0
- torchvision: 0.5.0

For more detailed environment settings, please refer to TCTSP/environment.yml

## Prepare the dataset

## Download the checkpoints

## Evaluate
To conduct evaluation of the pre-trained model, you can run the following commands:
```shell
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder /home/tangt/nfs_tangt/code/my-image-to-paragraph/experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test --resume 57 --markov_mat_path /home/tangt/nfs_tangt/code/my-image-to-paragraph/data/markov_mat_kmeans.npy
```

## Acknowledgement
