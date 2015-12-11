__author__ = 'Sun'

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np

import codecs


class NeuralResponseMachine(object):


    def build(self, corpus):


        graph = Graph()
        graph.add_input(name='src_sequence', ndim=1, dtype='int32')
        graph.add_input(name='tgt_sequence', ndim=1, dtype='int32')

        all_term_count = corpus.word_num()
        embedding_layer = Embedding(all_term_count, 256)
        graph.add_node(embedding_layer, name="src_embedding", input="src_sequence")
        graph.add_node(embedding_layer, name="tgt_embedding", input="tgt_sequence")

        graph.add_node(LSTM(256, 5120, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False),
                       name="src_encoder", input='src_embedding')


        # repmat & concat with tgt_embedding
        # send to decoder rnn to generate sequence





        graph.add_input(name='entity_1_uri', ndim=2, dtype='int32')
        graph.add_input(name='entity_2_term', ndim=2, dtype='int32')
        graph.add_input(name='entity_2_uri', ndim=2, dtype='int32')
        graph.add_input(name='entity_12_stat_sim', ndim=2)
        #graph.add_input(name='entity_12_kb_sim', ndim=2)

        graph.add_node(Embedding(all_term_count, 50), #, W_regularizer=l2(alpha)),
            name="entity_1_term_embedding", input="entity_1_term")
        graph.add_node(Convolution1D(50, 50, conv_window), #W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
         name="entity_1_term_conv", input="entity_1_term_embedding")
        graph.add_node(Activation('sigmoid'), name="entity_1_term_act", input="entity_1_term_conv")
        graph.add_node(MaxPooling1D(pool_length=pool_length), name="entity_1_term_pool", input="entity_1_term_act")
        graph.add_node(Flatten(), name="entity_1_term_flatten", input="entity_1_term_pool")

        graph.add_node(Embedding(all_uri_count, 50), #, W_regularizer=l2(alpha)),
             name="entity_1_uri_embedding", input="entity_1_uri")
        graph.add_node(Flatten(), name="entity_1_uri_flatten", input="entity_1_uri_embedding")

        graph.add_node(Embedding(all_term_count, 50), # W_regularizer=l2(alpha)),
             name="entity_2_term_embedding", input="entity_2_term")
        graph.add_node(Convolution1D(50, 50, conv_window), # W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
            name="entity_2_term_conv", input="entity_2_term_embedding")
        graph.add_node(Activation('sigmoid'), name="entity_2_term_act", input="entity_2_term_conv")
        graph.add_node(MaxPooling1D(pool_length=pool_length), name="entity_2_term_pool", input="entity_2_term_act")
        graph.add_node(Flatten(), name="entity_2_term_flatten", input="entity_2_term_pool")

        graph.add_node(Embedding(all_uri_count, 50), #W_regularizer=l2(alpha)),
            name="entity_2_uri_embedding", input="entity_2_uri")
        graph.add_node(Flatten(), name="entity_2_uri_flatten", input="entity_2_uri_embedding")

        graph.add_node(Dense(2*(50+50) + stat_sim_dim, 128), #W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
            name="entity_12_merge_trans1",
                       inputs=["entity_1_term_flatten", "entity_1_uri_flatten",
                               "entity_2_term_flatten", "entity_2_uri_flatten",
                               "entity_12_stat_sim"])

        graph.add_node(Activation('sigmoid'), name="entity_12_act1", input="entity_12_merge_trans1")
        graph.add_node(Dense(128, 128), # W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
            name="entity_12_trans2", input="entity_12_act1")
        graph.add_node(Activation('relu'), name="entity_12_act2", input="entity_12_trans2")
        graph.add_node(Dense(128, 128), # W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
             name="entity_12_trans3", input="entity_12_act2")
        graph.add_node(Activation('relu'), name="entity_12_act3", input="entity_12_trans3")
        graph.add_node(Dense(128, 1), # W_regularizer=l2(alpha), activity_regularizer=activity_l2(alpha)),
             name="entity_12_trans4", input="entity_12_act3")
        graph.add_node(Activation('sigmoid'), name="entity_12_act4", input="entity_12_trans4")

        graph.add_output(name='output', input='entity_12_act4')

        graph.compile('adagrad', {'output':'binary_crossentropy'})




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





