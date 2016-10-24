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

import logging

def to_time_distributed_categorical(y, nb_classes=None):

    sample_size, time_steps = y.shape
    if not nb_classes:
        nb_classes = np.max(y)+1

    Y = np.zeros((sample_size, time_steps, nb_classes))
    Y[np.arange(y.shape[0])[:, np.newaxis], np.arange(y.shape[1]), y] = 1

    return Y

def bucket_iter_adapter(bucket_iter, nb_classes):

    while True:
        #logging.debug("trying to read data")
        try:
            batch = bucket_iter.next()
        #    logging.debug("Batch generated")
            yield batch.data[0], to_time_distributed_categorical(batch.label[0].astype(np.int32), nb_classes)
        except StopIteration:
            bucket_iter.reset()

from keras.backend.common import _EPSILON
import theano.tensor as T
from theano import tensor as T, function, printing


def _debug_nan_fn(op, xin):

    if np.isnan(xin).any():
        logging.error("Nan detected in output")
        logging.error(str(xin))

def categorical_crossentropy(output, target):


    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    checking_output = printing.Print('Softmax', global_fn= _debug_nan_fn)(output)

    objective = -T.sum(target * T.log(checking_output),
                axis=checking_output.ndim - 1)

    return printing.Print('Objective', global_fn= _debug_nan_fn)(objective)


class SequenceTaggingMachine(object):

    def __init__(self):

        self.model = None


    def train(self, train_corpus, valid_corpus, learning_param):

        self.model = Sequential()

        self.model.add(Embedding(train_corpus.source_cell_num(), 256, mask_zero = True))

        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))

        self.model.add(TimeDistributed(Dense(input_dim=128, output_dim=train_corpus.target_cell_num())))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer='rmsprop',metrics=['accuracy'])


        logging.debug("Preparing data iter")
        problem = SequenceTaggingProblem(train_corpus)
        data_train = BucketIter(problem, learning_param.batch_size, max_pad_num=learning_param.max_pad)

        val_problem = SequenceTaggingProblem(valid_corpus)
        data_val = BucketIter(val_problem, learning_param.batch_size, max_pad_num=learning_param.max_pad)

	logging.debug("Begin train model")
        self.model.fit_generator(bucket_iter_adapter(data_train,train_corpus.target_cell_num()),
                                 samples_per_epoch=train_corpus.corpus_size(), nb_epoch=100, verbose=1,
                                 validation_data=bucket_iter_adapter(data_val, train_corpus.target_cell_num()),
                                 nb_val_samples = valid_corpus.corpus_size())

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
    logging.debug("Train corpus built")

    unlabeled_tag_id = train_corpus.target_corpus.id("U")

    val_corpus = train_corpus.make(codecs.open(validating_data, 'r', encoding="utf8"), segmenter, segmenter)
    logging.debug("Validate corpus built")

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





