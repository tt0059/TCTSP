import numpy as np
import json
import re
from collections import defaultdict

    
caps_dict = json.load(open('/home/huangyq/my-image-to-paragraph/Bert_clustering/Clustering-by-Silhouett/caps_dict.json')) # 83308个句子和对应的index
index_doc = caps_dict['index_doc'] # 83308个句子的index
caps_doc = caps_dict['caps_doc'] # 83308个句子

data = np.load('stanford_umap_embeddings_10.npy')
soft_path = '../../data_vg/topic_labels_soft_kmeans/'
hard_path = '../../data_vg/topic_labels_hard_kmeans/'

labels = np.load('umap_hard_labels_norm_kmeans_10.npy')
centers = np.load('umap_centers_norm_kmeans_10.npy')

num_samples = labels.shape[0]
label_list = list(set(list(labels)))
num_classes = len(label_list)
clu_result = defaultdict(list)

def compute_similarity(data):
    sim = np.zeros([1, num_classes])
    for i in range(num_classes):
        sim[0, i] = 2 / (1 - sum(data * centers[i]))
        #sim[0, i] = (1 - sum(data * centers[i]))/2 error but trained 29 cider,
    sim = sim / sum(sum(sim))
    return sim



for i in range(num_samples):
    clu_result[labels[i]].append(caps_doc[i])
for i in range(num_classes):
    print(len(clu_result[i]))
debug = 1


normed_data = data.copy()
for i in range(num_samples):
    normed_data[i] = normed_data[i] / np.linalg.norm(normed_data[i], 2)


i = 0
mis = 0
while i < len(index_doc):
    topic_label_hard = - np.ones([1,20])
    topic_label_soft = np.zeros([1,20, num_classes])
    image_id = re.split('\_',index_doc[i])[0]
    
    for j in range(len(index_doc) - i):
        cur_image_id = re.split('\_',index_doc[i+j])[0]
        #print(cur_image_id)
        if(cur_image_id != image_id):
            break
        #hard
        a = np.argmax(compute_similarity(normed_data[i+j]))
        topic_label_hard[0,j] = a #labels[i+j] #all sentences are clustered
        #soft
        topic_label_soft[0,j] = compute_similarity(normed_data[i+j]) #return the normalized simlarity of each clusters
      
        #a = np.argmax(compute_similarity(normed_data[i+j]))
        #if(a != labels[i+j]):
        #    mis += 1
          
    np.save(soft_path+image_id+'.npy', topic_label_soft)
    np.save(hard_path+image_id+'.npy', topic_label_hard)
    i += j
    print(i)
    if(i == len(index_doc)-1):
        break