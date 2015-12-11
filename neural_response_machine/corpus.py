__author__ = 'Sun'

import codecs
import numpy as np
import jieba

from collections import Counter


class NRMCorpus(object):
    def __init__(self):

        self.word_id_map = dict()
        self.id_word_map = dict()
        self.word_counter = Counter()

        self.corpus = []

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

        self.word_id_map = dict((word, id) for id, word in enumerate(pop_words))
        self.id_word_map = dict((id, word) for id, word in enumerate(pop_words))

        for src_words, tgt_words in raw_corpus:

            src_ids = [self.word_id_map[word] for word in src_words if word in self.word_id_map]
            tgt_ids = [self.word_id_map[word] for word in tgt_words if word in self.word_id_map]

            self.corpus.append((src_ids, tgt_ids))

        print "Corpus build."
        print "Word num = ", len(self.word_id_map)
        print "Corpus size = ", len(self.corpus)


    def get_id(self, char):
        return self.word_id_map[char]

    def get_char(self, id):
        return self.id_word_map[id]

    def word_num(self):
        return len(self.id_word_map)



