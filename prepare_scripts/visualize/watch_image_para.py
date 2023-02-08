import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import torch

imgs = json.load(open('data_paragraph/captions/para_karpathy_format.json'))['images']


for i,img in enumerate(imgs):
    ids = img['id']
    if(ids == 2360686):
        debug = 1
    caps = img['sentences'][0]['raw']
    #im_file = '/media/hyq/part2/datasets/visual_genome/images_cap/{}.jpg'.format(ids)
    #im = cv2.imread(im_file)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.axis('off')
    #plt.title(caps)
    #plt.imshow(im)   
    #plt.show()
    #debug = 1
    #plt.close()

    

plt.show()
debug = 1