__author__ = 'Sun'

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Reshape, RepeatVector, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
from neural_machine.tasks.language.sequence_tagging.sequence_tagging import SequenceTaggingProblem, LearnParam, ArchParam
from neural_machine.tasks.language.common.data_reader.bucket_iter import BucketIter

from neural_machine.tasks.language.common.corpus.segmentor import *
from neural_machine.tasks.language.common.corpus.sequence_corpus import SequenceCorpus
from neural_machine.tasks.language.common.corpus.sequence_pair_corpus import SequencePairCorpus
from neural_machine.tasks.language.common.data_reader.bucket_iter import *

def bucket_iter_adapter(bucket_iter):

    for batch in bucket_iter:
        yield batch.data[0].asnumpy(), batch.label[0].asnumpy()


class SequenceTaggingMachine(object):

    def __init__(self):

        self.model = None


    def train(self, train_corpus, valid_corpus, learning_param):

        self.model = Sequential()

        self.model.add(Embedding(train_corpus.source_cell_num(), 256, mask_zero = True))

        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))

        self.model.add(TimeDistributedDense(input_dim=128, output_dim=train_corpus.target_cell_num()))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


        print "Begin train model"
        problem = SequenceTaggingProblem(train_corpus)
        data_train = BucketIter(problem, learning_param.batch_size, max_pad_num=learning_param.max_pad)

        val_problem = SequenceTaggingProblem(valid_corpus)
        data_val = BucketIter(val_problem, learning_param.batch_size, max_pad_num=learning_param.max_pad)


        self.model.fit_generator(bucket_iter_adapter(data_train),
                                 batch_size=learning_param.batch_size, nb_epoch=100, verbose=2,
                                 validation_data=bucket_iter_adapter(data_val))

        print "Model is trained"


import logging
import codecs

import click

@click.command()
@click.argument("training_data")
@click.argument("validating_data")
@click.option("--batch_size", type=click.INT, default=100)
@click.option("--max_pad", type=click.INT, default=5)
def train_model(training_data, validating_data, batch_size, max_pad):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    segmenter = CharacterSegmenter()
    train_corpus = SequencePairCorpus(source_with_unk=True, same_length=True)

    train_corpus.build(codecs.open(training_data, 'r', encoding="utf8"), segmenter, segmenter)

    unlabeled_tag_id = train_corpus.target_corpus.id("U")

    val_corpus = train_corpus.make(codecs.open(validating_data, 'r', encoding="utf8"), segmenter, segmenter)


    learning_param = LearnParam(
        num_epoch=25, learning_rate=0.05, momentum=0.0,
        batch_size=batch_size,
        max_pad = max_pad, device=None, nworker=None
    )



    lm = SequenceTaggingMachine()


    logging.log(logging.INFO, "Begin to train ...")
    lm.train(train_corpus, val_corpus, learning_param)


if __name__ == "__main__":
    train_model()





