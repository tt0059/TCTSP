import numpy as np
import os
import pickle


hard_path = '/home/huangyq/my-image-to-paragraph/data_vg/topic_labels_hard_kmeans'
soft_path = '/home/huangyq/my-image-to-paragraph/data_vg/topic_labels_soft_kmeans'

listDir_hard = os.listdir(hard_path)
listDir_soft = os.listdir(soft_path)

# print(len(listDir_hard))
# print(len(listDir_soft))
# print(len(set(listDir_hard)))
# print(len(set(listDir_soft)))

# print(set(listDir_hard)==set(listDir_soft))

# num = 0
########################################计算topic hard label与topic soft label之间有哪些npy文件存在差异##########################
# for i in listDir_hard:
#     hard_npy = os.path.join(hard_path,i)
#     soft_npy = os.path.join(soft_path,i)
#     hard_npy_data = np.load(hard_npy)
#     soft_npy_data = np.load(soft_npy)
#     flag = 0
#     for j in range(20):
#         soft_index = np.where(soft_npy_data[0][j]==np.max(soft_npy_data[0][j]))[0][0]
#         hard_index = hard_npy_data[0][j]
#         if(soft_index!=hard_index):
#             flag = 1
#     if(flag==1):
#         num = num+1
#         print(i)

# print(num)
########################################验证第一次的topic gts与soft label对应，第二次topic gts和hard label对应####################
file4=open("/home/huangyq/my-image-to-paragraph/data_vg/vg_train_topic_gts_silhouett.pkl","rb")
file5=open("/home/huangyq/my-image-to-paragraph/data_vg/vg_train_topic_gts_silhouett2.pkl","rb")
image_ids = '/home/huangyq/my-image-to-paragraph/data_vg/train_ids.txt'

with open(image_ids) as f:
    image_ids = [line.strip() for line in f]

data4=pickle.load(file4)
data5=pickle.load(file5)

# print(list(data4[1][0]))
# print(list(np.maximum(data4[1][0],0)))
# print(list(data5[1][0]))
#############hard##########################
print("hard:")
num_hard=0
for i in range(len(image_ids)):# every image
    hard_npy = os.path.join(hard_path,image_ids[i]+'.npy')
    hard_npy_data = np.load(hard_npy)
    flag_hard = 0
    for j in range(20):
        hard_index = hard_npy_data[0][j]
        data5_index = list(data5[i][0])[j]
        if(hard_index!=data5_index):
            flag_hard = 1
    if(flag_hard==1):
        print(i)
        num_hard = num_hard+1
print("num_hard="+str(num_hard))

#############soft###########################
print("soft:")
num_soft=0
for i in range(len(image_ids)):# every image
    soft_npy = os.path.join(soft_path,image_ids[i]+'.npy')
    soft_npy_data = np.load(soft_npy)
    flag_soft = 0
    for j in range(20):
        soft_index = np.where(soft_npy_data[0][j]==np.max(soft_npy_data[0][j]))[0][0]
        data4_index = list(np.maximum(data4[i][0],0))[j]
        if(soft_index!=data4_index):
            flag_soft = 1
    if(flag_soft==1):
        print(i)
        num_soft = num_soft+1

print("num_soft="+str(num_soft))

file4.close()
file5.close()


# a = np.load('/home/huangyq/my-image-to-paragraph/data_vg/topic_labels_soft_kmeans/2387143.npy')
# print(len(a[0]))
# print(np.where(a[0][1]==np.max(a[0][1]))[0][0])

# b = np.load('/home/huangyq/my-image-to-paragraph/data_vg/topic_labels_hard_kmeans/2387143.npy')
# print(b[0])
# # a