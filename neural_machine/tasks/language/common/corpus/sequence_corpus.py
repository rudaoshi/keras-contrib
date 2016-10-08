__author__ = 'Sun'

import codecs
import numpy as np

import logging

class SequenceCorpus(object):
    def __init__(self):

        self.cell_id_map = dict()
        self.id_cell_map = dict()

        self.corpus = []

        # 0 is reserved for padding
        self.cell_id_map[""] = 0
        self.id_cell_map[0] = ""

#        self.cell_id_map["<start>"] = 1
#        self.id_cell_map[1] = "<start>"

        self.cell_id_map["<eos>"] = 1
        self.id_cell_map[1] = "<eos>"

        self.cell_id_map["<unk>"] = 2
        self.id_cell_map[2] = "<unk>"

    def build(self, data_file, segmentor):

        for line in data_file:

            cells = segmentor.segment(line)

            cur_curpus = [0] * (len(cells) + 2)
            cur_curpus[0] = 1
            cur_curpus[-1] = 2

            for idx in range(1, len(cells) + 1):
                cell = cells[idx-1]
                if cell not in self.cell_id_map:
                    id = len(self.cell_id_map)
                    self.cell_id_map[cell] = id
                    self.id_cell_map[id] = cell
                else:
                    id = self.cell_id_map[cell]

                cur_curpus[idx] = id

            self.corpus.append(cur_curpus)

        logging.info("Corpus build.")
        logging.info("Character num = {}".format(len(self.cell_id_map)))
        logging.info("Corpus size = {}".format(len(self.corpus)))

    def make(self, data_file, segmentor):

        corpus = SequenceCorpus()
        corpus.cell_id_map = self.cell_id_map
        corpus.id_cell_map = self.id_cell_map

        for line in data_file:

            cells = segmentor.segment(line)

            cur_curpus = [0] * (len(cells) + 2)
            cur_curpus[0] = 1
            cur_curpus[-1] = 2

            for idx in range(1, len(cells) + 1):
                cell = cells[idx-1]
                if cell not in self.cell_id_map:
                    id = self.id_cell_map["<unk>"]
                else:
                    id = self.cell_id_map[cell]

                cur_curpus[idx] = id

            corpus.corpus.append(cur_curpus)

        return corpus

    def id(self, cell):
        return self.cell_id_map[cell]

    def cell(self, id):
        return self.id_cell_map[id]

    def cell_num(self):
        return len(self.id_cell_map)

    def corpus_size(self):
        return len(self.corpus)



