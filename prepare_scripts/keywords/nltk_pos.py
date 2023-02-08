import nltk
import json
import re
pos_per_img = {}

imgs = json.load(open('data_vg/para_myfix_format.json'))['images']
max_sents = 0
max_len = 0
for i,img in enumerate(imgs):
    ids = img['id']
    caps = img['sentences'][0]['raw']
    ori_sent = re.sub('\.\.','.',caps)
    new_sent = ' . '.join(re.split('\.', ori_sent))
    
    caps_list = nltk.word_tokenize(new_sent)
    pos = nltk.pos_tag(caps_list)
    
    cur_img = {'1':[]}
    cur_sent = 1
    cur_num = 0
    for j, cur_pos in enumerate(pos):
        if(cur_pos[0] == '.' and j != len(pos)-1):
            sent_len = j - cur_num
            if(sent_len > 60):
                print(ids)
                print(new_sent)
                print(sent_len)
                print(img['split'])
                debug = 1
            if(sent_len > max_len):
                #if(sent_len > 40):
                #    print(max_len)
                #    debug = 1
                max_len = sent_len
                
            cur_num = j
            cur_sent += 1
            cur_img[str(cur_sent)] = []
            continue
        if(cur_pos[1] == 'JJ' or 'NN' in cur_pos[1] or cur_pos[1] == 'VBG' or cur_pos[1] == 'VBN'):
            cur_img[str(cur_sent)].append(cur_pos[0].lower())
    if(cur_sent > max_sents):
        max_sents = cur_sent
    pos_per_img[ids] = cur_img
    #debug = 1
    
    #debug = 1

json.dump(pos_per_img, open('./data_vg/keywords_per_img.json','w'))
print(max_sents)
print(max_len)