__author__ = 'Sun'

import codecs
import numpy as np
import jieba

from collections import Counter
from utils.data_process import to_time_distributed_categorical

class WordSequenceCorpus(object):
    def __init__(self):

        self.word_id_map = dict()
        self.id_word_map = dict()
        self.word_counter = Counter()

        self.corpus = dict()

    def build(self, data_file):

        raw_corpus = []
        for line in data_file:
            line = line.strip()
            if not line or "\t" not in line:
                continue

            src_sentence, tgt_sentence = line.split('\t')

            src_words = jieba.cut(src_sentence)
            tgt_words = jieba.cut(tgt_sentence)

            all_words = src_words + tgt_words

            self.word_counter.update(all_words)

            raw_corpus.append((src_words, tgt_words))

        pop_words = [word for word in self.word_counter if self.word_counter[word] >= 2]

        self.word_id_map = dict((word, id+1) for id, word in enumerate(pop_words))
        self.word_id_map["<EOS>"] = 0
        self.id_word_map = dict((id+1, word) for id, word in enumerate(pop_words))
        self.id_word_map[0] = "<EOS>"

        src_input = []
        #tgt_input = []
        tgt_output = []

        for src_words, tgt_words in raw_corpus:

            src_ids = [self.word_id_map[word] for word in src_words if word in self.word_id_map]
            src_ids.append(0)
            src_input.append(np.array(src_ids, dtype="int32"))

            tgt_ids = [self.word_id_map[word] for word in tgt_words if word in self.word_id_map]
            tgt_ids.append(0)
            #tgt_input.append(tgt_ids[:-1])

            tgt_output.append(np.array(tgt_ids, dtype="int32"))

        self.corpus = {
            "src_sequence": np.array(src_input, dtype="int32"),
            #"tgt_input": np.array(tgt_input, dtype="int32"),
            "tgt_sequence": np.array(tgt_output, dtype="int32")
        }

        print "Corpus build."
        print "Word num = ", len(self.word_id_map)
        print "Corpus size = ", self.corpus["src_input"].shape[0]


    def get_id(self, char):
        return self.word_id_map[char]

    def get_char(self, id):
        return self.id_word_map[id]

    def word_num(self):
        return len(self.id_word_map)

    def max_tgtseq_len(self):

        return max([len(x[1]) for x in self.corpus])

    def split(self, train_ratio):

        train_seq_num = train_ratio*self.corpus['src_input'].shape[0]
        X_train = self.corpus['src_input'][:train_seq_num]
        Y_train = self.corpus['tgt_output'][:train_seq_num]

        X_test = self.corpus['src_input'][train_seq_num:]
        Y_test = self.corpus['tgt_output'][:train_seq_num]

        train_corpus = WordSequenceCorpus()
        train_corpus.word_id_map = self.word_id_map
        train_corpus.id_word_map = self.id_word_map
        train_corpus.word_counter = self.word_counter
        train_corpus.corpus =  {
            "src_sequence": X_train,
            "tgt_sequence": Y_train
        }

        test_corpus = WordSequenceCorpus()
        test_corpus.word_id_map = self.word_id_map
        test_corpus.id_word_map = self.id_word_map
        test_corpus.word_counter = self.word_counter
        test_corpus.corpus =  {
            "src_sequence": X_test,
            "tgt_sequence": Y_test
        }

        return train_corpus, test_corpus


    def get_sequence_map(self, categorical_output = False):

        if categorical_output:
            output = to_time_distributed_categorical(self.corpus['tgt_output'], self.corpus.word_num())
        else:
            output = self.corpus['tgt_output']

        return self.corpus['src_input'], output








