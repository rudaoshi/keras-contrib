

import numpy as np
import mxnet as mx
from mxnet import ndarray
from mxnet.metric import EvalMetric, check_label_shapes

from neural_machine.tasks.language.common.problem import Problem
from neural_machine.component.lstm import StackedLSTM, BidirectionalStackedLSTM




class MaskedAccuracy(EvalMetric):
    """Calculate accuracy"""

    def __init__(self, mask):
        super(MaskedAccuracy, self).__init__('accuracy')
        self.mask = mask

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            pred_label = ndarray.argmax_channel(pred_label).asnumpy().astype('int32')
            label = label.reshape((label.size,)).asnumpy().astype('int32')

            check_label_shapes(label, pred_label)
            pred_label = pred_label[np.logical_and(label!=self.mask, label!=0)]
            label = label[np.logical_and(label!=self.mask ,label!=0)]

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

class SequenceTaggingProblem(Problem):

    def __init__(self, corpus):
        super(SequenceTaggingProblem, self).__init__(corpus)

    def is_supervised(self):
        return True

    def data_names(self):
        return ["data"]

    def label_names(self):

        return ["label"]

    def samples(self):

        for sample, label in self.corpus.corpus:
            yield [sample], [label]




class ArchParam(object):

    def __init__(self, num_hidden, num_embed,
                 num_lstm_layer, input_cell_num, output_cell_num):

        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.num_lstm_layer = num_lstm_layer
        self.input_cell_num = input_cell_num
        self.output_cell_num = output_cell_num

class LearnParam(object):

    def __init__(self, num_epoch, learning_rate, momentum,
        batch_size, device, nworker
    ):
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.device = device
        self.nworker = nworker

import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"

@mx.operator.register("masked_softmax")
class MaskedSoftmaxProp(mx.operator.CustomOpProp):

    def __init__(self, mask=None):
        super(MaskedSoftmaxProp, self).__init__(need_top_grad=False)
        self.mask = mask

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]


    def create_operator(self, ctx, shapes, dtypes):
        return MaskedSoftmax(self.mask)

class MaskedSoftmax(mx.operator.CustomOp):
    def __init__(self, mask):
        super(MaskedSoftmax, self).__init__()
        self.mask = mask


    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = mx.nd.exp(x - mx.nd.max(x, axis=1).reshape((x.shape[0], 1)))
        y = y/mx.nd.sum(y, axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], y)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()

        #logging.log(logging.DEBUG, "l = {0}, y = {1}".format(l.shape, y.shape))

        y[np.arange(l.shape[0]), l] -= 1.0
        y[l == self.mask, :] = 0.0
        y[l == 0, : ] = 0.0  # padding
        self.assign(in_grad[0], req[0], mx.nd.array(y))


class PartialLabeledSenquenceTaggingModel(object):

    def __init__(self, param, unlabeled_tag_id):

        self.param = param
        self.mask = unlabeled_tag_id


        #self.init_states = RepeatedAppendIter(
        #    [np.zeros((batch_size, self.param.num_hidden))] * 4,
        #    ['l{0}_init_{1}'.format(l, t)
        #     for l in range(self.param.num_lstm_layer)
        #     for t in ["c", "h"]])

    def __build(self, bucket):

        embed_weight = mx.sym.Variable("embed_weight")

        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")

        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        seq_len = bucket[0]

        embed = mx.sym.Embedding(data=data, input_dim=self.param.input_cell_num,
                                 weight=embed_weight, output_dim=self.param.num_embed, name='embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

        lstm = BidirectionalStackedLSTM(self.param.num_lstm_layer,
                           self.param.num_hidden, seq_len, return_sequence=True)(wordvec)

        pred = mx.sym.FullyConnected(data=lstm, num_hidden=self.param.output_cell_num,
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

        sm = mx.symbol.Custom(data=pred, label=label,  mask = self.mask, name='softmax', op_type='masked_softmax')

        #sm = MaskedSoftmax(self.mask)(data=pred, label=label, name='softmax')

        return sm


    def train(self, data_train, data_val, learning_param):

        self.symbol = lambda seq_len: self.__build(seq_len)

	state_data_names = ['{0}_l{1}_init_{2}'.format(direction, l, t)
             for l in range(self.param.num_lstm_layer)
             for t in ["c", "h"]
             for direction in ["forward", "backward"]]

        init_states = RepeatedAppendIter(
            [np.zeros((learning_param.batch_size, self.param.num_hidden))] * len(state_data_names),
            state_data_names)

        if learning_param.device == "cpu":
            contexts = [mx.context.cpu(i) for i in range(learning_param.nworker)]
        elif learning_param.device == "gpu":
            contexts = [mx.context.gpu(i) for i in range(learning_param.nworker)]

        self.model = mx.model.FeedForward(ctx=contexts,
                                          symbol=self.symbol,
                                          num_epoch=learning_param.num_epoch,
                                          learning_rate=learning_param.learning_rate,
                                          momentum=learning_param.momentum,
                                          wd=0.00001,
                                          initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

        """

        """



        train_iter = MergeIter(data_train, init_states)

        val_iter = None
        if data_val:
            val_iter = MergeIter(data_val, init_states)


        self.model.fit(X=train_iter, eval_data=val_iter,
                  eval_metric=MaskedAccuracy(self.mask),
                  batch_end_callback=mx.callback.Speedometer(learning_param.batch_size, 50), )


    def show_shape_info(self, data_train):

        train_iter = MergeIter(data_train, self.init_states)

        default_symbol = self.__build(train_iter.default_bucket_key)

        arg_shape, output_shape, aux_shape = default_symbol.infer_shape(
            **dict(train_iter.provide_data + train_iter.provide_label)
        )
        arg_names = default_symbol.list_arguments()
        aux_names = default_symbol.list_auxiliary_states()

        for i in range(len(arg_names)):
            print arg_names[i], arg_shape[i] if arg_shape is not None else "None"

        for i in range(len(aux_names)):
            print aux_names[i], aux_shape[i] if aux_shape is not None else "None"

        print "output shape", output_shape

from neural_machine.tasks.language.common.corpus.segmentor import *
from neural_machine.tasks.language.common.corpus.sequence_corpus import SequenceCorpus
from neural_machine.tasks.language.common.corpus.sequence_pair_corpus import SequencePairCorpus
from neural_machine.tasks.language.common.data_reader.bucket_iter import *

import sys

import logging

import codecs

import click

@click.command()
@click.argument("training_data")
@click.option("--batch_size", type=click.INT, default = 100)
@click.option("--max_pad", type=click.INT, default = 5)
@click.option("--dev", type=click.Choice(['gpu', 'cpu']), default="cpu")
@click.option("--nworker", type=click.INT, default=2)
def train_model(training_data, batch_size, max_pad, dev, nworker):

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    segmenter = CharacterSegmenter()
    corpus = SequencePairCorpus(source_with_unk=True, same_length=True)

    corpus.build(codecs.open(training_data, 'r', encoding = "utf8"), segmenter, segmenter)

    problem = SequenceTaggingProblem(corpus)

    data_train = BucketIter(problem, batch_size, max_pad_num = max_pad)

#    val_corpus = corpus.make(open(sys.argv[2], 'r'), segmenter)
#    val_problem = LanguageModelProblem(val_corpus)
#    data_val = BucketIter(val_problem, batch_size)


    arch_param = ArchParam(
        num_hidden= 200,
        num_embed= 200,
        num_lstm_layer= 6,
        input_cell_num = corpus.source_cell_num(),
        output_cell_num= corpus.target_cell_num()
    )

    learning_param = LearnParam(
        num_epoch=25,learning_rate=0.01, momentum=0.0,
        batch_size = batch_size, device=dev, nworker = nworker
    )

    unlabeled_tag_id = corpus.target_corpus.id("U")
    lm = PartialLabeledSenquenceTaggingModel(arch_param, unlabeled_tag_id)


    #lm.show_shape_info(data_train)

    logging.log(logging.INFO, "Begin to train ...")
    lm.train(data_train, None, learning_param)


if __name__ == "__main__":

    train_model()
