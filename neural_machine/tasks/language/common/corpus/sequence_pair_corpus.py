__author__ = 'Sun'

import codecs
import numpy as np
import jieba

import itertools
from collections import Counter
from neural_machine.tasks.language.common.corpus.sequence_corpus import SequenceCorpus
from utils.data_process import to_time_distributed_categorical

class SequencePairCorpus(object):
    def __init__(self,
                 source_with_start=False, source_with_end = False, source_with_unk = False,
                 target_with_start=False, target_with_end=False, target_with_unk=False,
                 ):

        self.source_corpus = SequenceCorpus(source_with_start, source_with_end, source_with_unk)
        self.target_corpus = SequenceCorpus(target_with_start, target_with_end, target_with_unk)

        self.corpus = []

    def build(self, data_file, source_segmenter, target_segmenter):

        for line in data_file:
            line = line.strip()
            if not line or "\t" not in line:
                continue

            src_seq, tgt_seq = line.split('\t')

            self.source_corpus.update(src_seq, source_segmenter)
            self.target_corpus.update(tgt_seq, target_segmenter)

        for src, target in itertools.izip(self.source_corpus.corpus, self.target_corpus.corpus):
            self.corpus.append((src, target))



    def source_cell_num(self):
        return self.source_corpus.cell_num()

    def target_cell_num(self):
        return self.target_corpus.cell_num()

    def corpus_size(self):
        return len(self.corpus)








