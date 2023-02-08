import json

objects = open('/media/hyq/part2/datasets/visual_genome/objects_vocab.txt').readlines()
ix_to_obj = {}
for i in range(len(objects)):
    ix_to_obj[i+1] = objects[i]


attrs = open('/media/hyq/part2/datasets/visual_genome/attributes_vocab.txt').readlines()
ix_to_attr = {}
for i in range(len(attrs)):
    ix_to_attr[i+1] = attrs[i]

infos = {'ix_to_obj':ix_to_obj, 'ix_to_attr':ix_to_attr}

json.dump(infos, open('/media/hyq/part2/datasets/visual_genome/vocab_all.json','w'))