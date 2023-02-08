import numpy as np
import json
import re

caps_dict = json.load(open('caps_dict.json'))
index_doc = caps_dict['index_doc']
caps_doc = caps_dict['caps_doc']
clu_result = {}

soft_path = '../../data_vg/topic_labels_10_5_soft_silhouett/'
hard_path = '../../data_vg/topic_labels_10_5_hard_silhouett/'

labels = np.load('umap_hard_labels_norm.npy')
probs = np.load('umap_probs_norm.npy')

num_samples = labels.shape[0]
#label_list = []
#for i in range(num_samples):
#    label_list.append(labels[i])
label_list = list(set(list(labels)))
debug = 1

for i in range(len(label_list)):
    clu_result[label_list[i]] = [] #

for i in range(num_samples):
    clu_result[labels[i]].append(caps_doc[i])
debug = 1
    


i = 0
while i < len(index_doc):
    topic_label_hard = np.zeros([1,20])
    topic_label_soft = np.zeros([1,20, len(label_list)])
    image_id = re.split('\_',index_doc[i])[0]
    
    for j in range(len(index_doc) - i):
        cur_image_id = re.split('\_',index_doc[i+j])[0]
        #print(cur_image_id)
        if(cur_image_id != image_id):
            break
        #hard
        topic_label_hard[0,j] = labels[i+j]+1  #put -1 to 0

        #soft
        
        if(labels[i+j] == -1):
            prob = 0.5
        else:
            prob = probs[i+j]
        other_avg = (1 - prob)/(len(label_list) - 1)
        topic_label_soft[0,j] = other_avg
        topic_label_soft[0, j, labels[i+j]+1] = prob

        
        
              
    np.save(soft_path+image_id+'.npy', topic_label_soft)
    np.save(hard_path+image_id+'.npy', topic_label_hard)
    i += j
    print(i)
    if(i == len(index_doc)-1):
        break