__author__ = 'Sun'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np

import codecs

from sandbox.dynamic_title.creator.char_corpus import CharacterCorpus


def to_time_distributed_categorical(y, nb_classes=None):

    sample_size, time_steps = y.shape
    if not nb_classes:
        nb_classes = np.max(y)+1

    Y = np.zeros((sample_size, time_steps, nb_classes))

    for i in range(sample_size):
        for j in range(time_steps):
            Y[i, j, y[i, j]] = 1.
    return Y

def train_rnn(character_corpus, seq_len, train_test_split_ratio):
    model = Sequential()
    model.add(Embedding(character_corpus.char_num(), 256))
    model.add(LSTM(256, 5120, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributedDense(5120, character_corpus.char_num()))
    model.add(Activation('time_distributed_softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    seq_X, seq_Y = character_corpus.make_sequences(seq_len)

    print "Sequences are made"

    train_seq_num = train_test_split_ratio*seq_X.shape[0]
    X_train = seq_X[:train_seq_num]
    Y_train = to_time_distributed_categorical(seq_Y[:train_seq_num], character_corpus.char_num())

    X_test = seq_X[train_seq_num:]
    Y_test = to_time_distributed_categorical(seq_Y[train_seq_num:], character_corpus.char_num())

    print "Begin train model"
    checkpointer = ModelCheckpoint(filepath="model.step", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=256, nb_epoch=100, verbose=2, validation_data=(X_test, Y_test), callbacks=[checkpointer])

    print "Model is trained"

    score = model.evaluate(X_test, Y_test, batch_size=512)

    print "valid score = ", score

    return model

import cPickle
import click
@click.command()
@click.argument("char_cropus_file", type=click.File(mode='rb'))
@click.argument("model_file", type=click.File(mode='wb'))
@click.option("--seq_len", type=click.INT, default=5)
@click.option("--split_ratio", type=click.FLOAT, default=0.8)
def train_char_rnn(char_cropus_file, model_file,
                    seq_len, split_ratio):

    corpus = cPickle.load(char_cropus_file)

    model = train_rnn(corpus, seq_len, split_ratio)

    cPickle.dump(model, model_file, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    train_char_rnn()





