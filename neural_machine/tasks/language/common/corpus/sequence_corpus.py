__author__ = 'Sun'

import codecs
import numpy as np

class SequenceCorpus(object):
    def __init__(self):

        self.cell_id_map = dict()
        self.id_cell_map = dict()

        self.corpus = []
        self.cell_id_map["<start>"] = 0
        self.id_cell_map[0] = "<start>"

        self.cell_id_map["<eos>"] = 1
        self.id_cell_map[1] = "<eos>"

    def build(self, data_file, segmentor):

        for line in data_file:

            cells = segmentor.segment(line)

            cur_curpus = [0] * (len(cells) + 2)
            cur_curpus[0] = 0
            cur_curpus[-1] = 1

            for idx in range(1, len(cells) + 1):
                cell = cells[idx]
                if cell not in self.cell_id_map:
                    id = len(self.cell_id_map)
                    self.cell_id_map[cell] = id
                    self.id_cell_map[id] = cell
                else:
                    id = self.cell_id_map[cell]

                cur_curpus[idx] = id

            self.corpus.extend(cur_curpus)

        print "Corpus build."
        print "Character num = ", len(self.cell_id_map)
        print "Corpus size = ", len(self.corpus)


    def id(self, cell):
        return self.cell_id_map[cell]

    def cell(self, id):
        return self.id_cell_map[id]

    def cell_num(self):
        return len(self.id_cell_map)

    def corpus_size(self):
        return len(self.corpus)

    def make_sequences(self, seq_length):

        if len(self.corpus) % seq_length != 0:
            print "cutting of the end of data"

        seq_num = len(self.corpus) / seq_length
        x = self.corpus[: seq_num * seq_length]

        y = x[1:]; y.append(x[0])

        seq_X = np.array(x).reshape((seq_num, seq_length))
        seq_Y = np.array(y).reshape((seq_num, seq_length))

        return (seq_X, seq_Y)

