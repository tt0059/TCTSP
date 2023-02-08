import pickle
import numpy as np
image_ids = '/home/huangyq/my-image-to-paragraph/data_vg/train_ids.txt'
gts = []
with open(image_ids) as f:
    image_ids = [line.strip() for line in f]

for _ in range(len(image_ids)):
    gts.append([])

for i, image_id in enumerate(image_ids):
    seqs = np.load('/home/huangyq/my-image-to-paragraph/data_vg/topic_labels_hard_kmeans/'+image_id+'.npy').astype('int')[0]
    gts[i].append(seqs)
pickle.dump(gts, open('/home/huangyq/my-image-to-paragraph/data_vg/vg_train_topic_gts_silhouett2.pkl', 'wb'))
