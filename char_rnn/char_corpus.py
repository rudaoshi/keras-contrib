__author__ = 'Sun'

import codecs
import numpy as np

class CharacterCorpus(object):
    def __init__(self):

        self.char_id_map = dict()
        self.id_char_map = dict()

        self.corpus = []

    def build(self, data_file):

        for line in data_file:
            cur_curpus = [0] * len(line)
            for idx, char in enumerate(line):
                if char not in self.char_id_map:
                    id = len(self.char_id_map)
                    self.char_id_map[char] = id
                    self.id_char_map[id] = char
                else:
                    id = self.char_id_map[char]

                cur_curpus[idx] = id

            self.corpus.extend(cur_curpus)

        print "Corpus build."
        print "Character num = ", len(self.char_id_map)
        print "Corpus size = ", len(self.corpus)


    def get_id(self, char):
        return self.char_id_map[char]

    def get_char(self, id):
        return self.id_char_map[id]

    def char_num(self):
        return len(self.id_char_map)

    def make_sequences(self, seq_length):

        if len(self.corpus) % seq_length != 0:
            print "cutting of the end of data"

        seq_num = len(self.corpus) / seq_length
        x = self.corpus[: seq_num * seq_length]

        y = x[1:]; y.append(x[0])

        seq_X = np.array(x).reshape((seq_num, seq_length))
        seq_Y = np.array(y).reshape((seq_num, seq_length))

        return (seq_X, seq_Y)

