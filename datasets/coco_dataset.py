import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
#from tools.tokenization import BertTokenizer, WordTokenizer

class CocoDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, 
        input_seq, 
        target_seq,
        att_feats_folder, 
        seq_per_img,
        max_feat_num,
        topic_label_folder,
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.image_ids = utils.load_lines(image_ids_path) # 与image_id_path顺序一致
        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        #self.gv_feat = None#pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None
        self.topic_label_folder = topic_label_folder
        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = 175
        else:
            self.input_seq = None
            self.target_seq = None
            self.seq_len = -1

       # self.tokenizer = WordTokenizer("./data_vg/vg_vocab.txt")
         
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        indices = np.array([index]).astype('int')
        
        
        #if self.gv_feat is not None:
            #gv_feat = self.gv_feat[image_id]
            #gv_feat = np.array(gv_feat).astype('float32')
        #else:
            #gv_feat = np.zeros((1,1))

        if self.att_feats_folder is not None:
            att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['x']
            #np.savez_compressed(output_file, x=image_feat, bbox=image_bboxes, num_bbox=len(keep_boxes), image_h=np.size(im, 0), image_w=np.size(im, 1), info=info)
            if(len(att_feats.shape) == 3):
                att_feats = att_feats.reshape(-1, 2048)
            att_feats = np.array(att_feats).astype('float32')
        else:
            att_feats = np.zeros((1,1))
        
            
        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
            att_feats = att_feats[:self.max_feat_num, :]
       
        attn_labels = np.load('/home/tangt/nfs_tangt/code/my-image-to-paragraph/data_vg/attn_labels/'+str(image_id)+'.npy')    
        
        if self.seq_len < 0:
            topic_labels = np.zeros([1,20]).astype(int)
            return indices, image_id, att_feats, topic_labels 
        else:
            topic_labels = np.load(os.path.join(self.topic_label_folder,str(image_id)+'.npy')).astype(np.float32) 
            if(topic_labels.shape[-1] == 20):
                topic_labels = topic_labels.astype(int)
            #print(topic_labels)
        
        

        input_seq =  self.input_seq[image_id][np.newaxis,:].astype(int)
        target_seq = self.target_seq[image_id][np.newaxis,:].astype(int)
        
        
        
        return indices, image_id, input_seq, target_seq, att_feats, attn_labels, topic_labels