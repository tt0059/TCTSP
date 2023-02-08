CUDA_VISIBLE_DEVICES=6 \
python \
/home/tangt/nfs_tangt/code/TCTSP/main_test.py \
--folder /home/tangt/nfs_tangt/code/TCTSP/experiments/Xlan_SAP_V6_kmeans_wt03_RL_wt05_CIDEr_25_test \
--resume 57 \
--markov_mat_path /home/tangt/nfs_tangt/code/TCTSP/data/markov_mat_kmeans.npy \
--topic_num 82 \