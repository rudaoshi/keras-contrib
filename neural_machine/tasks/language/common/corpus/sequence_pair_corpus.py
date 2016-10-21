__author__ = 'Sun'

import codecs
import numpy as np
import json, logging
import itertools
from collections import Counter
from neural_machine.tasks.language.common.corpus.sequence_corpus import SequenceCorpus
from utils.data_process import to_time_distributed_categorical

class SequencePairCorpus(object):
    def __init__(self,
                 source_with_start=False, source_with_end = False, source_with_unk = False,
                 target_with_start=False, target_with_end=False, target_with_unk=False,
                 same_length = False
                 ):

        self.source_corpus = SequenceCorpus(source_with_start, source_with_end, source_with_unk)
        self.target_corpus = SequenceCorpus(target_with_start, target_with_end, target_with_unk)
        self.same_length = same_length

        self.corpus = []

    def build(self, data_file, source_segmenter, target_segmenter):

        for line in data_file:
            line = line.strip()
            if not line:
                continue

            try:
                src_seq, tgt_seq = line.split('\t')
            except:
                logging.error("no sequence pair found in sentence : {0} ".format(json.dumps(line)))
                continue

            if self.same_length and len(src_seq) != len(tgt_seq):
                logging.error("src and tgt seq not in same length {0} {1} {2}".format(len(src_seq), len(tgt_seq), json.dumps(line)))
                continue

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








