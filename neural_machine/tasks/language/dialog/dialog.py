

import numpy as np
import mxnet as mx

from neural_machine.tasks.language.common.problem import Problem
from neural_machine.component.lstm import StackedLSTM, SequenceDecoder

class DialogProblem(Problem):

    def __init__(self, corpus):
        super(DialogProblem, self).__init__(corpus)

    def is_supervised(self):
        return True

    def data_names(self):
        return ["data"]

    def label_names(self):

        return ["label"]

    def samples(self):

        for x in self.corpus.corpus:
            data = x[:-1]
            label = x[1:]
            yield [data], [label]

    @staticmethod
    def objective(label, pred):
        "Perplexity for language model"

        #logging.debug("{0} {1}".format(label.shape, pred.shape))

        label = label.T.reshape((-1,))
        loss = 0.
        for i in range(pred.shape[0]):
            try:
                loss += -np.log(max(1e-10, pred[i][int(label[i])]))
            except:
                print >> sys.stderr, pred
                print >> sys.stderr, label
                raise

        return np.exp(loss / label.size)


class DialogModelArchParam(object):

    def __init__(self, num_hidden, num_embed,
                 num_lstm_layer, cell_num):

        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.num_lstm_layer = num_lstm_layer
        self.cell_num = cell_num

class DialogModelLearnParam(object):

    def __init__(self, num_epoch, learning_rate, momentum):
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum

class DialogModel(object):

    def __init__(self, param):

        self.param = param

    def __build(self, bucket):

        embed_weight = mx.sym.Variable("embed_weight")

        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")

        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        seq_len = bucket[0]

        embed = mx.sym.Embedding(data=data, input_dim=self.param.cell_num,
                                 weight=embed_weight, output_dim=self.param.num_embed, name='embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

        encoder_out, states = StackedLSTM(self.param.num_lstm_layer,
                           self.param.num_hidden, seq_len, name = "encoder",
                              return_sequence=False,
                              output_states=True)(wordvec)

        decoder_out = SequenceDecoder(self.param.num_lstm_layer,
                           self.param.num_hidden, seq_len, name = "decoder",
                              init_states=states,
                              return_sequence=True,
                              output_states=False)(encoder_out)

        pred = mx.sym.FullyConnected(data=decoder_out, num_hidden=self.param.cell_num,
                                     weight=cls_weight, bias=cls_bias, name='pred')

        ################################################################################
        # Make label the same shape as our produced data path
        # I did not observe big speed difference between the following two ways

        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, shape=(-1,))

        # label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
        # label = [label_slice[t] for t in range(seq_len)]
        # label = mx.sym.Concat(*label, dim=0)
        # label = mx.sym.Reshape(data=label, target_shape=(0,))
        ################################################################################

        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return sm


    def train(self, data_train, data_val, learning_param):

        self.symbol = lambda seq_len: self.__build(seq_len)

        contexts = [mx.context.cpu(i) for i in range(1)]
        self.model = mx.model.FeedForward(ctx=contexts,
                                          symbol=self.symbol,
                                          num_epoch=learning_param.num_epoch,
                                          learning_rate=learning_param.learning_rate,
                                          momentum=learning_param.momentum,
                                          wd=0.00001,
                                          initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

        init_states = RepeatedAppendIter(
            [np.zeros((batch_size, self.param.num_hidden))] * 4,
            ["encoder" + 'l{0}_init_{1}'.format(l, t)
             for l in range(self.param.num_lstm_layer)
             for t in ["c", "h"]])

        train_iter = MergeIter(data_train, init_states)
        val_iter = MergeIter(data_val, init_states)

        print train_iter.provide_data

        self.model.fit(X=train_iter, eval_data=val_iter,
                  eval_metric=mx.metric.np(DialogProblem.objective),
                  batch_end_callback=mx.callback.Speedometer(batch_size, 50), )


    def show_shape_info(self, train_iter):

        default_symbol = self.symbol(train_iter.default_bucket_key)

        arg_shape, output_shape, aux_shape = default_symbol.infer_shape(
            **dict(train_iter.provide_data + train_iter.provide_label)
        )
        arg_names = default_symbol.list_arguments()
        aux_names = default_symbol.list_auxiliary_states()

        for i in range(len(arg_names)):
            print arg_names[i], arg_shape[i]

        for i in range(len(aux_names)):
            print aux_names[i], aux_shape[i]

        print "output shape", output_shape

from neural_machine.tasks.language.common.corpus.segmentor import *
from neural_machine.tasks.language.common.corpus.sequence_corpus import SequenceCorpus
from neural_machine.tasks.language.common.data_reader.bucket_iter import *

import sys

import logging

if __name__ == '__main__':

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


    segmenter = SpaceSegmenter()
    corpus = SequenceCorpus()

    corpus.build(open(sys.argv[1], 'r'), segmenter)
    cell_num = corpus.cell_num()

    problem = DialogProblem(corpus)

    batch_size = 32

    data_train = BucketIter(problem, batch_size)

    val_corpus = corpus.make(open(sys.argv[2], 'r'), segmenter)
    val_problem = DialogProblem(val_corpus)
    data_val = BucketIter(val_problem, batch_size)


    arch_param = DialogModelArchParam(
        num_hidden= 200,
        num_embed= 200,
        num_lstm_layer= 2,
        cell_num = corpus.cell_num()
    )

    learning_param = DialogModelLearnParam(
        num_epoch=25,learning_rate=0.01, momentum=0.0
    )

    lm = DialogModel(arch_param)

    lm.train(data_train, data_val, learning_param)

