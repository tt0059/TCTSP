CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
--nproc_per_node=1 \
/home/tangt/my-image-to-paragraph/main_soft_train.py \
--folder /home/tangt/my-image-to-paragraph/experiments/Xlan_SAP_V6_kmeans_real_vinvl_wt03_RL_wt05/ \
--resume 0 \
--topic_weight 0.05 \
--tensorboard_path /home/tangt/my-image-to-paragraph/experiments/Xlan_SAP_V6_kmeans_real_vinvl_wt03_RL_wt05/log \
--markov_mat_path /nfs/tangt/code/my-image-to-paragraph/data_vg/markov_mat_kmeans.npy \
--topic_num 82 \
