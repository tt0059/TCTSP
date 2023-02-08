from losses.cross_entropy import CrossEntropy
from losses.label_smoothing import LabelSmoothing
from losses.reward_criterion import RewardCriterion
from losses.reward_criterion_topic import RewardCriterionTopic
from losses.KL_loss import KLLoss
from losses.supervised_attn import SuperAttn
from losses.sup_feature import SuperFeat
from losses.RangeShrink_sum import RangeShrink_sum
from losses.slide_window_sum import slide_window_sum
from losses.CaLCS import CaLCS
__factory = {
    'CrossEntropy': CrossEntropy,
    'LabelSmoothing': LabelSmoothing,
    'RewardCriterion': RewardCriterion,
    'RewardCriterionTopic': RewardCriterionTopic,
    'KLLoss': KLLoss,
    'SuperAttn': SuperAttn,
    'SuperFeat': SuperFeat,
    'RangeShrink_sum': RangeShrink_sum,
    'slide_window_sum': slide_window_sum,
    'CaLCS': CaLCS,
}

def names():
    return sorted(__factory.keys())

def create(name,*args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)