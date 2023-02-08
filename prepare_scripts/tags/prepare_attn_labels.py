import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import torch


imgs = json.load(open('data_vg/para_myfix_format.json'))['images']
keywords = json.load(open('./data_vg/keywords_per_img.json'))

infos = json.load(open('/media/hyq/part2/datasets/visual_genome/vocab_all.json'))
classes = infos['ix_to_obj']
attributes = infos['ix_to_attr']

save_root = './data_vg/keywords_labels/'


#sent_attr_labels = []

#np.savez_compressed(output_file, x=image_feat, bbox=image_bboxes, num_bbox=len(keep_boxes), image_h=np.size(im, 0), image_w=np.size(im, 1), info=info)


for i, img in enumerate(imgs):
    ids = img['id']   
    try:
        tp = np.load('/media/hyq/part2/datasets/visual_genome/res101_10_100_ray/feature/{}.npz'.format(ids),allow_pickle=True)
        a = tp['info']
        boxes = tp['bbox']
        objects = a.item()['objects_id']
        scores = a.item()['objects_conf']
        attrs = a.item()['attrs_id']
        attr_scores = a.item()['attrs_conf']
        
        im_file = '/media/hyq/part2/datasets/visual_genome/images_cap/{}.jpg'.format(ids)
        
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        
    
        
        keywords_cur = keywords[str(ids)]
        
        labels = np.zeros([len(boxes),20])
        for sent_id in range(1, 1 + len(keywords_cur.keys())):
            gts = keywords_cur[str(sent_id)]
            num_box = 0
            #plt.axis('off')
            #plt.imshow(im)        
            for j in range(len(boxes)):
                bbox = boxes[j]
                if bbox[0] == 0:
                    bbox[0] = 1
                if bbox[1] == 0:
                    bbox[1] = 1
                    
                cls = classes[str(objects[j]+1)].strip()
                attr = attributes[str(attrs[j]+1)].strip()
                merged = attr + ' ' + cls
                if(cls in gts or attr in gts):  
                    labels[j, sent_id-1] = 1
                    #num_box += 1
                    #plt.gca().add_patch(
                        #plt.Rectangle((bbox[0], bbox[1]),
                                      #bbox[2] - bbox[0],
                                      #bbox[3] - bbox[1], fill=False,
                                      #edgecolor='red', linewidth=2, alpha=0.5)
                            #)
                    #plt.gca().text(bbox[0], bbox[1] - 2,
                                #'%s' % (merged),
                                #bbox=dict(facecolor='blue', alpha=0.5),
                                #fontsize=10, color='white')
                    
            
            #print('boxes={}'.format(num_box))
            #plt.xlabel(str(sent_id) + ' ' + str(gts))
            #plt.show()
            debug = 1
            #plt.close()
        np.save(save_root + str(ids)+'.npy', labels)
        debug = 1
        if(i%100 == 0):
            print(i)
    except:
        print(ids)