import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
#from datasets.coco_dataset_clip import CocoDataset
import samplers.distributed
import numpy as np

def sample_collate(batch):
    indices, image_id, input_seq, target_seq,  att_feats, attn_labels, topic_labels = zip(*batch)

    indices = np.stack(indices, axis=0).reshape(-1)
    image_id = np.stack(image_id, axis=0).reshape(-1)
    
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)    
    topic_labels = torch.cat([torch.from_numpy(b) for b in topic_labels], 0)


    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)



    attn_labels_arr = []
    for i, num in enumerate(atts_num):
        tmp_label = np.zeros((1, max_att_num, attn_labels[i].shape[1]), dtype=np.float32)
        tmp_label[:, 0:attn_labels[i].shape[0], :] = attn_labels[i]
        attn_labels_arr.append(torch.from_numpy(tmp_label))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    attn_labels = torch.cat(attn_labels_arr, 0)    

    #topic_labels_arr = []
    #for i, num in enumerate(atts_num):
        #tmp_label = np.zeros((1, 20, topic_labels[i].shape[1]), dtype=np.float32)
        #tmp_label[:, 0:topic_labels[i].shape[0], :] = topic_labels[i]
        #topic_labels_arr.append(torch.from_numpy(tmp_label))

        #tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        #tmp_mask[:, 0:num] = 1
        #mask_arr.append(torch.from_numpy(tmp_mask))

    #topic_labels = torch.cat(topic_labels_arr, 0)  
    
    return indices, image_id, input_seq, target_seq, att_feats, att_mask, attn_labels, topic_labels

def sample_collate_val(batch):
    indices, image_id, att_feats, topic_labels = zip(*batch)

    indices = np.stack(indices, axis=0).reshape(-1)
    image_id = np.stack(image_id, axis=0).reshape(-1)
    topic_labels = torch.cat([torch.from_numpy(b) for b in topic_labels], 0)

    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    
    #topic_labels_arr = []
    #for i, num in enumerate(atts_num):
        #tmp_label = np.zeros((1, 20, topic_labels[i].shape[1]), dtype=np.float32)
        #tmp_label[:, 0:topic_labels[i].shape[0], :] = topic_labels[i]
        #topic_labels_arr.append(torch.from_numpy(tmp_label))

        #tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        #tmp_mask[:, 0:num] = 1
        #mask_arr.append(torch.from_numpy(tmp_mask))

    #topic_labels = torch.cat(topic_labels_arr, 0)  
    
    return indices, image_id, att_feats, att_mask, topic_labels


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, att_feats_folder):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, 
        target_seq = None, 
        att_feats_folder = att_feats_folder,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
        topic_label_folder=cfg.DATA_LOADER.TOPIC_LABEL_PATH
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader