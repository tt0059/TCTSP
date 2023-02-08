import numpy as np
import json
import re
from collections import defaultdict

    
caps_dict = json.load(open('caps_dict.json'))
index_doc = caps_dict['index_doc']
caps_doc = caps_dict['caps_doc']

data = np.load('stanford_umap_embeddings_10.npy')


labels = np.load('umap_hard_labels_norm_kmeans_10.npy')
centers = np.load('umap_centers_norm_kmeans_10.npy')

num_samples = labels.shape[0]
label_list = list(set(list(labels)))
num_classes = len(label_list)
clu_result = defaultdict(list)



for i in range(num_samples):
    clu_result[labels[i]].append(caps_doc[i])
for i in range(num_classes):
    print(len(clu_result[i]))
debug = 1


