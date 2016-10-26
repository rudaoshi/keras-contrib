__author__ = 'Sun'

import codecs
import numpy as np

import logging
import copy

class SequenceCorpus(object):
    def __init__(self, with_start = False, with_end = False, with_unk = False, with_pad=True):

        self.cell_id_map = dict()
        self.id_cell_map = dict()

        self.with_start = with_start
        self.with_end = with_end
        self.with_unk = with_unk

        self.corpus = []

        if with_pad:
            # 0 is reserved for padding
            self.cell_id_map[""] = 0
            self.id_cell_map[0] = ""

        if with_start:
            id = len(self.cell_id_map)
            self.cell_id_map["<start>"] = id
            self.id_cell_map[id] = "<start>"

        if with_end:
            id = len(self.cell_id_map)
            self.cell_id_map["<eos>"] = id
            self.id_cell_map[id] = "<eos>"

        if with_unk:
            id = len(self.cell_id_map)
            self.cell_id_map["<unk>"] = id
            self.id_cell_map[id] = "<unk>"

    def update(self, seq, segmentor):

        cells = segmentor.segment(seq)

        cur_corpus = [0] * len(cells)

        for idx in range(len(cells)):
            cell = cells[idx]
            if cell not in self.cell_id_map:
                id = len(self.cell_id_map)
                self.cell_id_map[cell] = id
                self.id_cell_map[id] = cell
            else:
                id = self.cell_id_map[cell]

            cur_corpus[idx] = id

        if self.with_start:
            cur_corpus = [self.id("<start>")] + cur_corpus
        if self.with_end:
            cur_corpus = cur_corpus + [self.id("<eos>")]

        #self.corpus.append(cur_corpus)

        return cur_corpus


    def predict(self, seq, segmentor):

        cells = segmentor.segment(seq)

        cur_corpus = [0] * len(cells)

        for idx in range(len(cells)):
            cell = cells[idx]
            if cell not in self.cell_id_map:
                if self.with_unk:
                    id = self.id("<unk>")
                else:
                    raise Exception("Unknown cell found. the repo should build with with_unk = True")
            else:
                id = self.cell_id_map[cell]

            cur_corpus[idx] = id

        if self.with_start:
            cur_corpus = [self.id("<start>")] + cur_corpus
        if self.with_end:
            cur_corpus = cur_corpus + [self.id("<eos>")]

        return cur_corpus


    def build(self, data_file, segmentor):

        for line in data_file:

            cur_corpus = self.update(line, segmentor)
            self.corpus.append(cur_corpus)



        logging.info("Corpus build.")
        logging.info("Character num = {}".format(len(self.cell_id_map)))
        logging.info("Corpus size = {}".format(len(self.corpus)))


    def make(self, data_file, segmentor):

        corpus = SequenceCorpus(self.with_start, self.with_end, self.with_unk)
        corpus.cell_id_map = self.cell_id_map
        corpus.id_cell_map = self.id_cell_map

        for line in data_file:
            cur_curpus = self.predict(line, segmentor)

            corpus.corpus.append(cur_curpus)

        return corpus

    def clone(self):

        corpus = SequenceCorpus(self.with_start, self.with_end, self.with_unk)
        corpus.cell_id_map = copy.copy(self.cell_id_map)
        corpus.id_cell_map = copy.copy(self.id_cell_map)

        return corpus


    def id(self, cell):
        return self.cell_id_map[cell]

    def cell(self, id):
        return self.id_cell_map[id]

    def cell_num(self):
        return len(self.id_cell_map)

    def corpus_size(self):
        return len(self.corpus)



