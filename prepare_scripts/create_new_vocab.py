import json

vg_vocab = json.load(open('../data_vg/paratalk_filtered.json'))

ix_to_word = vg_vocab['ix_to_word']
ix_to_word[str(len(ix_to_word)+1)] = '.'


vocab = {'word_to_ix':vg_vocab['word_to_ix'],'ix_to_word':ix_to_word}
json.dump(vocab,open('../data_vg/vocab_filtered.json','w'))

with open('../data_vg/vg_vocab_filtered.txt','w') as f: #vocab_len 11730
    for i in range(1,len(ix_to_word)+1):
        word = ix_to_word[str(i)]+'\n'
        f.write(word)
        