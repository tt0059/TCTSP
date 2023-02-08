import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile
        ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        self.is_eval = eval_annfile is not None
        if(self.is_eval):
            self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)


    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask, topic_labels):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.TOPIC_LABELS] = topic_labels
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname, model_type):
        model.eval()

        results = [] # 存储模型在验证集/测试集上的预测结果
        with torch.no_grad():
            for _, (indices, image_id, att_feats, att_mask, topic_labels) in tqdm.tqdm(enumerate(self.eval_loader)): # every batch
                ids = self.eval_ids[indices]
                gv_feat = None
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                topic_labels = topic_labels.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask, topic_labels)
                if kwargs['BEAM_SIZE'] > 1:
                    if(model_type == 'UpDown' or model_type == 'Att2inModel'):
                        seq, _ = model.module.decode_beam_slow(**kwargs)
                    else:
                        seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _, topic_sents, _ = model.module.decode(**kwargs) # 当前的评估方法暂且没有使用到topic_sents
                sents = utils.decode_sequence(self.vocab, seq.data) # 对预测的句子由数字映射到单词或标点（貌似标点只有句号）
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent, 'topic sequence': topic_sents[sid][topic_sents[sid]!=-1].tolist()} # 这里记录一下topic
                    results.append(result)
                #break

        if(self.is_eval):
            eval_res = self.evaler.eval(results) # 对生成结果进行评估

            result_folder = os.path.join(cfg.ROOT_DIR, 'result')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

            model.train()
            return eval_res, topic_sents
        else:
            return None