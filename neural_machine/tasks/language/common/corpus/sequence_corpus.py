__author__ = 'Sun'

import codecs
import numpy as np

import logging
import copy

from neural_machine.tasks.language.common.corpus.cell_dict import CellDict


class SequenceCorpus(object):


    def __init__(self, with_start = False, with_end = False, with_unk = False, with_pad=True):

        self.with_start = with_start
        self.with_end = with_end
        self.with_unk = with_unk

        self.cell_dict = CellDict(with_start, with_end, with_unk, with_pad)
        self.corpus = []

    def update(self, seq, segmentor):

        cells = segmentor.segment(seq)

        return self.cell_dict.update(cells)


    def predict(self, seq, segmentor):

        cells = segmentor.segment(seq)

        return self.cell_dict.update(cells)


    def build(self, data_file, segmentor):

        for line in data_file:

            cur_corpus = self.update(line, segmentor)
            self.corpus.append(cur_corpus)

        logging.info("Corpus build.")
        logging.info("Cell num = {}".format(self.cell_dict.cell_num()))
        logging.info("Corpus size = {}".format(len(self.corpus)))


    def make(self, data_file, segmentor):

        corpus = SequenceCorpus(self.with_start, self.with_end, self.with_unk)
        corpus.cell_dict = copy.deepcopy(self.cell_dict)

        for line in data_file:
            cur_curpus = corpus.predict(line, segmentor)

            corpus.corpus.append(cur_curpus)

        return corpus

    def clone(self):

        return copy.deepcopy(self)

    def iter_ids(self):

        return self.cell_dict.iter_ids()


    def iter_cell(self):

        return self.cell_dict.iter_cell()

    def id(self, cell):
        return self.cell_dict.id(cell)

    def cell(self, id):
        return self.cell_dict.cell(id)

    def cell_num(self):
        return self.cell_dict.cell_num()

    def corpus_size(self):
        return len(self.corpus)



