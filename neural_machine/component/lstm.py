import mxnet as mx

import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

import logging
class StackedLSTM(object):


    def __init__(self, num_layer, num_hidden, seq_len,
                 name = "",
                 init_states = None,
                 return_sequence=True,
                 output_states = False):

        self.num_layer = num_layer
        self.seq_len = seq_len
        self.num_hidden = num_hidden

        self.param_cells = []
        self.last_states = []
        for i in range(num_layer):
            self.param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(name + "l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable(name + "l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable(name + "l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable(name + "l%d_h2h_bias" % i)))

            if init_states is None:
                state = LSTMState(c=mx.sym.Variable(name + "l%d_init_c" % i),
                              h=mx.sym.Variable(name + "l%d_init_h" % i))
            else:
                state = LSTMState(c=init_states.c,
                              h=init_states.h)

            self.last_states.append(state)

        self.return_sequence = return_sequence
        self.output_states = output_states
        self.name = name



    def step(self, data, seqidx, layeridx):

        param = self.param_cells[layeridx]
        prev_state = self.last_states[layeridx]

        i2h = mx.sym.FullyConnected(data=data,
                                    weight=param.i2h_weight,
                                    bias=param.i2h_bias,
                                    num_hidden=self.num_hidden * 4,
                                    name=self.name + "t%d_l%d_i2h" % (seqidx, layeridx))
        h2h = mx.sym.FullyConnected(data=prev_state.h,
                                    weight=param.h2h_weight,
                                    bias=param.h2h_bias,
                                    num_hidden=self.num_hidden * 4,
                                    name=self.name+"t%d_l%d_h2h" % (seqidx, layeridx))
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name=self.name+"t%d_l%d_slice" % (seqidx, layeridx))
        in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
        in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
        forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
        next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
        return LSTMState(c=next_c, h=next_h)


    def __call__(self, data):

        hidden_all = []
        for seqidx in range(self.seq_len):
            hidden = data[seqidx]

            # stack LSTM
            for i in range(self.num_layer):
                next_state = self.step(hidden, seqidx, i)
                hidden = next_state.h
                self.last_states[i] = next_state

            hidden_all.append(hidden)

        if self.return_sequence:
            hidden_concat = mx.sym.Concat(*hidden_all, dim=0)

            output_data =  hidden_concat
        else:
            output_data = hidden_all[-1]

        if self.output_states:
            return output_data, self.last_states[-1]


class SequenceDecoder(StackedLSTM):

    def __init__(self, * args, ** kwargs):
        super(SequenceDecoder, self).__init__(*args, **kwargs)

    def __call__(self, data):

        hidden_all = []
        input = data
        for seqidx in range(self.seq_len):
            # stack LSTM
            for i in range(self.num_layer):
                next_state = self.step(input, seqidx, i)
                hidden = next_state.h
                self.last_states[i] = next_state

            hidden_all.append(hidden)
            input = hidden

        if self.return_sequence:
            hidden_concat = mx.sym.Concat(*hidden_all, dim=0)

            output_data = hidden_concat
        else:
            output_data = hidden_all[-1]

        if self.output_states:
            return output_data, self.last_states[-1]
        else:
            return output_data