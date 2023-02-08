import pickle
import torch

# i=2

image_ids = '/home/huangyq/my-image-to-paragraph/data_vg/train_ids.txt'
with open(image_ids) as f:
    image_ids = [line.strip() for line in f]

file1=open("/home/huangyq/my-image-to-paragraph/data_vg/vg_train_gts.pkl","rb")
file2=open("/home/huangyq/my-image-to-paragraph/data_vg/vg_train_gts_silhouett.pkl","rb")
file3=open("/home/huangyq/my-image-to-paragraph/data_vg/vg_train_target.pkl",'rb')
data1=pickle.load(file1)
data2=pickle.load(file2)
data3=pickle.load(file3)
# print(torch.tensor(data1[i]))
# print(torch.tensor(data2[i]))
# print(torch.tensor(data1[i])==torch.tensor(data2[i]))

# print(len(data1))
# print(len(data2))
# print([1,2,3]==[1,2,3])

num=0
for i in range(len(image_ids)):
    if(bool(1-(list(data3[image_ids[i]][0:len(data2[i][0])])==data2[i][0]))):
        print(i)
        num = num+1

print(num)
# print(bool(1-(list(data3[image_ids[i]][0:len(data1[i][0])])==data2[i][0])))

file1.close()
file2.close()
file3.close()