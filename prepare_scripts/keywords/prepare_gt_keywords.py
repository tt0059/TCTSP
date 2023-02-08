import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import torch


imgs = json.load(open('data_vg/para_myfix_format.json'))['images']
keywords = json.load(open('./data_vg/keywords_per_img.json'))
lines = open('/media/hyq/part2/my-image-to-paragraph/data_vg/vg_vocab.txt').readlines()
save_root = './data_vg/keywords/'
word_to_ix = {}
for i,line in enumerate(lines):
    word_to_ix[line.strip()] = i+1


max_num = 0

for i, img in enumerate(imgs):
    ids = img['id']   

    keywords_labels = np.zeros([20,20])
    keywords_cur = keywords[str(ids)]
    
    if(len(keywords_cur) > max_num):
        max_num = len(keywords_cur)
        if(max_num == 20):
            debug = 1
    for j in range(len(keywords_cur)):
        index = str(j+1)
        cnt = 0
        for k, word in enumerate(keywords_cur[index]):
            try:
                keywords_labels[j,cnt] = word_to_ix[word]   
                cnt += 1
            except:
                pass
                #print('unk!')
            if(cnt == 20):
                break            
    np.save(save_root + str(ids)+'.npy', keywords_labels)

    if(i%100 == 0):
        print(i)
print(max_num)