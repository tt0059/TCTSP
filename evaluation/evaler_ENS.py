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
        i3d_feat,
        res152_feats,
        eval_annfile
        ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, i3d_feat, res152_feats)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)

    def make_kwargs(self, indices, ids, i3d_feat, i3d_mask, res152_feats, res152_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.I3D_FEAT] = i3d_feat
        kwargs[cfg.PARAM.I3D_FEAT_MASK] = i3d_mask
        kwargs[cfg.PARAM.RES152_FEATS] = res152_feats
        kwargs[cfg.PARAM.RES152_FEATS_MASK] = res152_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        model.eval()

        results = []
        with torch.no_grad():
            for _, (indices, i3d_feat, i3d_mask, res152_feats, res152_mask) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = self.eval_ids[indices]
                i3d_feat = i3d_feat.cuda()
                i3d_mask = i3d_mask.cuda()
                res152_feats = res152_feats.cuda()
                res152_mask = res152_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, i3d_feat, i3d_mask, res152_feats, res152_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.decode_beam(**kwargs)
                else:
                    seq, _ = model.decode(**kwargs)
                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)
        if (len(results) == 3000):
            return None,results
        else:
            eval_res = self.evaler.eval(results)
    
            result_folder = os.path.join(cfg.ROOT_DIR, 'result')
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)
            json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))
    
            model.train()
            return eval_res,results