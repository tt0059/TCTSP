import json
import re
images = json.load(open('../data_vg/para_karpathy_format.json'))['images']

cur_set = 'val'
#imgs = [x for x in imgs if x['id'] not in [2346046, 2341671]]
data_dict = {'annotations':[], 'images':[]}
for img in images:
    split = img['split']
    if(split == cur_set):
        img_id = img['id']
        if(img_id == 2346046 or img_id == 2341671):
            continue
        caption = img['sentences'][0]['raw']
        #cur_sent = re.sub('\.  \.','.',caption)
        #cur_sent = re.sub('\. \.','.',caption)
        #cur_sent = re.sub('\.\.','.',caption)
        cur_sent = '. '.join(re.split('\.', caption))
        
        data_dict['annotations'].append({'id':img_id, 'image_id':img_id, 'caption':cur_sent})
        data_dict['images'].append({'id':img_id})
json.dump(data_dict,open('../data_vg/{}_2k_real.json'.format(cur_set),'w')) 
