import os
import sys
import json
import h5py
import numpy as np
import argparse
import pickle
from collections import defaultdict

def precook(words, n=4, out=False):# 这里貌似是在计算当前句子ngram频次
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average" 返回当前句子所有ngram频次
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

def remove_ignore(seq): # 去除句子填充
    sent = []
    for word in seq:
        if word == -1:
            break
        sent.append(word)
    return sent

def main(args):
    gts = []
    with open(args.image_ids) as f:
        image_ids = [line.strip() for line in f] # 训练集的image ids

    for _ in range(len(image_ids)): # 根据image ids个数创建gts列表
        gts.append([])

    target_seqs = pickle.load(open(args.infile, 'rb'), encoding='bytes') # 训练集中的所有gts
    for i, image_id in enumerate(image_ids): # 按照data_vg/train_ids.txt的id顺序填充gts
        seqs = np.array(target_seqs[image_id]).astype('int')
        gts[i].append(remove_ignore(seqs))
    # pickle.dump(gts, open(args.gts, 'wb')) #!!非调试请不要注释这一行

    crefs = [] # 计算所有段落ngram频次，列表中每一个元素对应一个段落
    for gt in gts:  # 按照image id次序存储每个image的n-gram的出现次数
        crefs.append(cook_refs(gt)) #看到了这里

    document_frequency = defaultdict(float)
    for refs in crefs: # 统计所有段落中每个ngram在多少个段落中出现过
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
            document_frequency[ngram] += 1
    ref_len = np.log(float(len(crefs))) # 段落个数取log 
    # pickle.dump({ 'document_frequency': document_frequency, 'ref_len': ref_len }, open(args.outfile, 'wb')) #!!非调试请不要注释这一行


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input h5
    parser.add_argument('--infile', default='../data_vg/vg_train_target.pkl', help='pkl file', type=str)
    parser.add_argument('--outfile', default='../data_vg/train_cider.pkl', help='output pickle file', type=str)
    parser.add_argument('--gts', default='../data_vg/vg_train_gts.pkl', help='output pickle file', type=str)
    parser.add_argument('--image_ids', default='../data_vg/train_ids.txt', help='image id file', type=str)

    args = parser.parse_args()
    main(args)
