import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.rouge_RL import Rouge
factory = {
    'Rouge': Rouge
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == -1:   # 这里原先是0，改成了-1，因为topic decode是以-1作为填充，不知道xe用不用改
            break
    return words

class Topic_Scorer(object):
    def __init__(self):
        super(Topic_Scorer, self).__init__()
        self.scorers = []
        path = cfg.SCORER.TOPIC_GT_PATH
        self.scorer = factory['Rouge']()
        self.gts = pickle.load(open(path, 'rb'), encoding='bytes')

    def __call__(self, ids, res):
        hypo = [get_sents(r) for r in res] # 排除句子中的填充，其实这里没起到作用，但是看起来和gts结构相同
        gts = [self.gts[i] for i in ids] # 获取gts  #!!感觉这里有问题啊,没问题了ids是一个index，是使用/home/huangyq/my-image-to-paragraph/data_vg/train_ids.txt作为列表的索引值，而非图像id

        rewards_info = {}
        rewards = np.zeros(len(ids))
        score, scores, places = self.scorer.compute_score(gts, hypo)
        rewards = np.array(scores)
        rewards_info['Weighted Rouge-L'] = score
        return rewards, rewards_info, places
