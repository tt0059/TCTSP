import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.cider_RL import Cider
from scorer.bleu_RL import Bleu
from scorer.meteor_RL import Meteor
factory = {
    'CIDEr': Cider,
    'Bleu': Bleu,
    'Meteor': Meteor
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

class Scorer(object):
    def __init__(self, name_list=['CIDEr'], weights = [1.0]):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = weights#cfg.SCORER.WEIGHTS
        self.scorer_type = []
        #if(name_list[0] == 'Rouge'):
        #    path = cfg.SCORER.TOPIC_GT_PATH#'./data_vg/vg_train_topic_gts.pkl'
        #else:
        path = cfg.SCORER.GT_PATH
        self.gts = pickle.load(open(path, 'rb'), encoding='bytes') # 硬标签
        for name in name_list:
            self.scorers.append(factory[name]())
            self.scorer_type.append(name)

    def __call__(self, ids, res):
        hypo = [get_sents(r) for r in res] # 这里是将句子最后的填充0去除，长短不一， 这里的candidate的情况应该是对标gts的
        gts = [self.gts[i] for i in ids] # 这里是在获取预测句子对应的gts，长短不一, 这里的gts是包含句号<eos>但不包含逗号

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers): 
            score, scores = scorer.compute_score(gts, hypo) # 计算每句的分数组成scores，之后求平均得到score(但bleu貌似不是这样的,Bleu是取最大值)
            if(self.scorer_type[i] == 'Bleu'):
                rewards += self.weights[i] * np.array(scores[3]) # 好像是如果计算Bleu就拿Bleu——4
            else:
                rewards += self.weights[i] * np.array(scores)
            rewards_info[self.scorer_type[i]] = score
        rewards = rewards / sum(self.weights) # 计算平均
        return rewards, rewards_info # rewards是每句话的分数都有记录，而rewards_info是记录了整个batch