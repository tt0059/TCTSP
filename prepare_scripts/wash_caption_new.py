import json
import re



myfile = json.load(open('../data_vg/para_filtered_format.json'))


cnt_tv = 0
cnt_website = 0
cnt_st = 0
cnt_misc = 0
cnt_banner = 0
cnt_dot = 0
cnt_punc_replace = 0
cnt_gang = 0
for i,img in enumerate(myfile['images']):
    raw_cap = img['sentences'][0]['raw'].lower()
    #filter tv
    if('\\' in raw_cap):
        raw_cap = raw_cap.replace('\\','')
        cnt_gang += 1        
        
    if('t.v' in raw_cap):
        raw_cap = re.sub('t\.v', 'tv', raw_cap)
        cnt_tv += 1
    
    if('.com' in raw_cap):
        raw_cap = re.sub('\.com', '@com', raw_cap)
        cnt_website += 1
    if('.org' in raw_cap):
        raw_cap = re.sub('\.org', '@org', raw_cap)
        cnt_website += 1            
    
    if(' st.' in raw_cap):
        raw_cap = re.sub(' st\.', ' street', raw_cap)
        cnt_st += 1    
    if(' dr.' in raw_cap):
        raw_cap = re.sub(' dr\.', ' doctor', raw_cap)
        cnt_misc += 1     
    if(' a.c' in raw_cap):
        raw_cap = re.sub(' a\.c', ' ac', raw_cap)
        cnt_misc += 1 
    if(' u.s' in raw_cap):
        raw_cap = re.sub(' u\.s', ' usa', raw_cap)
        cnt_misc += 1       
    if(' c.e.t' in raw_cap):
        raw_cap = re.sub(' c\.e\.t', ' cet', raw_cap)
        cnt_misc += 1        
      
    if('1.25' in raw_cap):
        raw_cap = re.sub('1\.25', '1@25', raw_cap)
        cnt_misc += 1     
    if('1.00' in raw_cap):
        raw_cap = re.sub('1\.00', '1@00', raw_cap)
        cnt_misc += 1          
       

    
    if('."' in raw_cap):
        raw_cap = re.sub('\."','"\.',raw_cap)
        cnt_punc_replace += 1
    if(re.search('".*?"', raw_cap) is not None):# and '.' in re.search('".*?"', raw_cap).group() and len(re.search('".*?"', raw_cap).group()) < 200):
        res = re.findall('".*?"', raw_cap)
        for j in range(len(res)):
            if len(res[j])<200 and '.' in res[j]:
                print(res[j])
                raw_cap = raw_cap.replace(res[j],'BANNER')
                cnt_banner += 1   
    
    
    if('...' in raw_cap):
        raw_cap = re.sub('\.\.\.', '.', raw_cap)
        cnt_dot += 1
    if('..' in raw_cap):
        raw_cap = re.sub('\.\.', '.', raw_cap)
        cnt_dot += 1    
    if('. .' in raw_cap):
        raw_cap = re.sub('\. \.', '.', raw_cap)
        cnt_dot += 1     
    
    
    img['sentences'][0]['raw'] = raw_cap
    if(img['id'] == 2373793):
        debug = 1
print(f'cnt_gang:{cnt_gang}')
print(f'cnt_tv:{cnt_tv}')
print(f'cnt_website:{cnt_website}')
print(f'cnt_st:{cnt_st}')
print(f'cnt_misc:{cnt_misc}')
print(f'cnt_punc_replace:{cnt_punc_replace}')
print(f'cnt_banner:{cnt_banner}')
print(f'cnt_dot:{cnt_dot}')

json.dump(myfile,open('../data_vg/para_filtered_v2_format.json','w'))