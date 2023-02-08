import csv
import numpy as np

#feat_num = 100
data = np.load('stanford_roberta_embeddings.npy')#[:feat_num]
headers = []
for i in range(data.shape[1]):
    headers.append('a'+str(i))
rows = []
for i in range(data.shape[0]):
    cur_data = data[i]
    cur_data = cur_data / np.linalg.norm(cur_data,2) # 当前embedding除以embedding的模
    #print(np.linalg.norm(cur_data))
    tp = cur_data.tolist() # 转化为列表
    rows.append(tp)

with open('umap_data_norm_2.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
    
    