import pickle
import json
import numpy as np
import re
import h5py
captions = h5py.File('../data_vg/paratalk_filtered_label.h5', 'r', driver='core')['labels']
all_images = json.load(open('../data_vg/paratalk_filtered.json'))['images']
train_txt = open('../data_vg/train_ids.txt','w')
val_txt = open('../data_vg/val_ids.txt','w')
test_txt = open('../data_vg/test_ids.txt','w')


encoded_num = 0
fail_num = 0
input_dict_train = {}
target_dict_train = {}
input_dict_val = {}
for i in range(len(all_images)):
    cur_image = all_images[i]
    image_id = str(cur_image['id'])
    split = cur_image['split']
    caps = captions[i].astype(np.int)
    last = np.argwhere(caps==2)[-1]
    caps[last] = 0#6153
    caps_tar = caps.copy()
    caps_tar[last.item()+1:] = -1
    if(split == 'train'):
        train_txt.write(str(image_id)+'\n')
        target_dict_train[image_id] = caps_tar
        caps_input = np.zeros_like(caps)
        caps_input[1:] = caps[:-1]
        caps_input[0] = 6111#6128
        input_dict_train[image_id] = caps_input
        #print(caps.shape)
        #debug = 1
    elif(split == 'val'):
        val_txt.write(str(image_id)+'\n')
        input_dict_val[image_id] = caps_tar
    else:
        test_txt.write(str(image_id)+'\n')
 
        
pickle.dump(input_dict_train,open('../data_vg/vg_train_input.pkl','wb'))
pickle.dump(target_dict_train,open('../data_vg/vg_train_target.pkl','wb'))
pickle.dump(input_dict_val,open('../data_vg/vg_val_input.pkl','wb'))


train_txt.close()
val_txt.close()
test_txt.close()
