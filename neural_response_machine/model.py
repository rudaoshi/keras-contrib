__author__ = 'Sun'

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Reshape, RepeatVector, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np





class GlobalNeuralRespondingMachine(object):

    def __init__(self, corpus):

        self.corpus = corpus
        self.model = None

    def build(self):

        self.model = Graph()
        self.model.add_input(name='src_sequence', input_shape=(None,1), dtype='int32')
        self.model.add_input(name='tgt_sequence', input_shape=(None,1), dtype='int32')

        all_term_count = self.corpus.word_num()
        embedding_layer = Embedding(all_term_count, 256)
        self.model.add_shared_node(embedding_layer, name="embedding",
                              inputs=["src_sequence","tgt_sequence"],
                              outputs=["src_embedding", "tgt_embedding"])

        self.model.add_node(GRU(input_dim=256, output_dim=128, return_sequences=False),
                       name="src_encoder", input='src_embedding')

        self.model.add_node(GRU(input_dim=128, output_dim=128, return_sequences=True),
                       name="decoder_gru", inputs=["src_encoder", "tgt_embedding"] )
        self.model.add_node(TimeDistributedDense(input_dim=128, output_dim=self.corpus.word_num()),
                       name="decoder_trans", input="decoder_gru" )
        self.model.add_node(Activation('time_distributed_softmax'),
                       name="decoder_active", input="decoder_trans" )

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



    def train(self):

        seq_X, seq_Y = self.corpus.get_sequence_map()

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

        # "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
        # "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
        # containing word index sequences representing partial captions.
        # "next_words" is a numpy float array of shape (nb_samples, vocab_size)
        # containing a categorical encoding (0s and 1s) of the next word in the corresponding
        # partial caption.
        self.model.fit([images, partial_captions], next_words, batch_size=16, nb_epoch=100)


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





