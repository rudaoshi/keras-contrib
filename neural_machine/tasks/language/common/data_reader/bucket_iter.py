import numpy as np
import mxnet as mx


# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

from collections import Counter
from operator import itemgetter


class UnsupervisedBatch(object):
    """Batch used for model parallelism"""

    def __init__(self, data, data_names, pad, index, bucket_key):
        self.data = [mx.nd.array(x) for x in data]
        self.data_names = data_names
        self.bucket_key = bucket_key
        self.pad = pad
        self.index = index


class SupervisedBatch(object):
    def __init__(self, data, data_names, label, label_names,
                 pad, index,
                 bucket_key):
        self.data = [mx.nd.array(x) for x in data]
        self.label = [mx.nd.array(x) for x in label]
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key
        self.pad = pad
        self.index = index

    @property
    def provide_data(self):
        return [(self.data_names[i], self.data[i].shape)
                for i in range(len(self.data_names))]

    @property
    def provide_label(self):
        return [(self.label_names[i], self.label[i].shape)
                for i in range(len(self.label_names))]


class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"

    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        self.default_bucket_key = real_iter.default_bucket_key

        for batch in real_iter:
            self.the_batch = batch
            if batch.pad == 0:
                break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


from mxnet.io import DataBatch

def pad(l, bucket_size):

    data = []

    for j in range(len(bucket_size)):
        np_array = np.zeros((len(l[j]), bucket_size[j]))
        for i in range(len(l[j])):
            np_array[i, :len(l[j][i])] = l[j][i]

        data.append(np_array)
    return data

import logging
class BucketIter(mx.io.DataIter):


    def gen_buckets(self, batch_size):
        shape_cap_map = Counter()

        for sample in self.problem.samples():
            shape = self.sample_shape(sample)

            shape_cap_map[shape] += 1

        bucket_capacity = sorted(shape_cap_map.iteritems(), key=lambda x: sum(np.array(x[0])))

        max_bucket = tuple(np.max(np.array(shape_cap_map.keys()),axis=0))
        tl = 0
        buckets = []
        for bucket, cap in bucket_capacity:  # TODO: There are better heuristic ways to do this
            logging.info("Buckets: {0} {1}".format(bucket, cap))
            if cap + tl >= batch_size:
                buckets.append(bucket)
                tl = 0
            else:
                tl += cap
        if tl > 0:
            buckets.append(max_bucket)

        return buckets, max_bucket

    def sample_shape(self, sample):

        if self.supervised:
            shape = [len(x) for x in sample[0]] + [len(x) for x in sample[1]]
        else:
            shape = [len(x) for x in sample]

        return tuple(shape)

    def __init__(self, problem, batch_size):
        super(BucketIter, self).__init__()

        self.problem = problem
        self.supervised = self.problem.is_supervised()

        self.data_names = problem.data_names()
        self.label_names = problem.label_names()

        self.buckets, self.default_bucket_key = self.gen_buckets(batch_size)

        self.data = [[[] for _ in range(len(self.data_names))]
                     for _ in range(len(self.buckets))]

        if self.supervised:
            self.label = [[[] for _ in range(len(self.data_names))]
                          for _ in range(len(self.buckets))]

        for sample in self.problem.samples():

            shape = self.sample_shape(sample)

            for i, bkt in enumerate(self.buckets):
                if np.all(np.array(bkt) >= np.array(shape)):

                    if self.supervised:
                        for j in range(len(self.data_names)):
                            self.data[i][j].append(sample[0][j])
                        for j in range(len(self.label_names)):
                            self.label[i][j].append(sample[1][j])
                    else:
                        for j in range(len(self.data_names)):
                            self.data[i][j].append(sample[j])
                    #logging.debug("bkt:{0}, shape:{1}".format(bkt, shape))

                    break
                    # we just ignore the sentence it is longer than the maximum
                    # bucket size here


        for i in range(len(self.buckets)):

            self.data[i] = pad(self.data[i],
                               self.buckets[i][0:len(self.data_names)])

            if self.supervised:
                self.label[i] = pad(self.label[i],
                                    self.buckets[i][len(self.data_names):])


        self.batch_size = batch_size

        self.shuffled_bucket_idx = np.random.permutation(len(self.data))
        self.cursor = (0,-1)

        self.provide_data = [(self.data_names[i], (self.batch_size, self.default_bucket_key[i]))
                             for i in range(len(self.data_names))]

        self.provide_label = [(self.label_names[i] , (self.batch_size, self.default_bucket_key[len(self.data_names) + i]))
                              for i in range(len(self.label_names))]

    def reset(self):
        """Reset the iterator. """
        self.shuffled_bucket_idx = np.random.permutation(len(self.data))
        self.cursor = (0, -1)

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)
        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():


            if self.supervised:
                return SupervisedBatch(data=self.getdata(),
                                       data_names = self.get_data_names(),
                                       label=self.getlabel(),
                                       label_names = self.get_label_names(),
                                       pad=self.getpad(),
                                       index=self.getindex(),
                                       bucket_key=self.get_bucket(),
                    )
            else:
                return UnsupervisedBatch(data=self.getdata(),
                                       data_names = self.get_data_names(),
                                       pad=self.getpad(),
                                       index=self.getindex(),
                                       bucket_key= self.get_bucket())
        else:
            raise StopIteration

    def get_data_names(self):

        return self.data_names

    def get_label_names(self):

        return self.label_names

    def get_bucket(self):

        i, j = self.cursor
        bucket_id = self.shuffled_bucket_idx[i]

        return self.buckets[bucket_id]


    def iter_next(self):
        """Iterate to next batch.
        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """

        i, j = self.cursor

        if j * self.batch_size >= self.data[self.shuffled_bucket_idx[i]][0].shape[0]:
            i += 1
            j = 0
        else:

            j += 1

        self.cursor = (i,j)

        if i >= len(self.shuffled_bucket_idx):
            return False

        return True


    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """

        i, j = self.cursor
        bucket_id = self.shuffled_bucket_idx[i]

        if (j+1) * self.batch_size < self.data[bucket_id][0].shape[0]:
            return [data[j * self.batch_size: (j+1) * self.batch_size, :]
                    for data in self.data[bucket_id]]

        else:
            cur_data = [data[j * self.batch_size:, :] for data in self.data[bucket_id]]
            return [np.vstack((x,
                             np.zeros((self.batch_size-x.shape[0],
                                       x.shape[1])))) for x in cur_data]

    def getlabel(self):
        """Get label of current batch.
        Returns
        -------
        label : NDArray
            The label of current batch.
        """

        i, j = self.cursor
        bucket_id = self.shuffled_bucket_idx[i]

        if (j+1) * self.batch_size < self.label[bucket_id][0].shape[0]:
            return [label[j * self.batch_size: (j+1) * self.batch_size, :]
                    for label in self.label[bucket_id]]

        else:
            cur_label = [label[j * self.batch_size:, :] for label in self.label[bucket_id]]
            return [np.vstack((x,
                             np.zeros((self.batch_size-x.shape[0],
                                       x.shape[1])))) for x in cur_label]

    def getindex(self):
        """
        Retures
        -------
        index : numpy.array
            The index of current batch
        """
        return np.zeros((self.batch_size,))

    def getpad(self):
        """Get the number of padding examples in current batch.
        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        i, j = self.cursor
        bucket_id = self.shuffled_bucket_idx[i]

        if (j + 1) * self.batch_size < len(self.label[bucket_id]):
            return 0
        else:
            return self.batch_size - len(self.label[bucket_id])% self.batch_size


class RepeatedAppendIter(mx.io.DataIter):

    def __init__(self, data, data_names):
        super(RepeatedAppendIter, self).__init__()

        self.data_names = data_names
        self.batch_size = data[0].shape[0]
        self.data = data

        self.default_bucket_key = None

        self.provide_data = [(self.data_names[i], self.data[i].shape) for i in range(len(data_names)) ]

    def get_data_names(self):

        return self.data_names

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)
        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():


            return UnsupervisedBatch(data=self.getdata(),
                                     data_names=self.data_names,
                                     pad=self.getpad(),
                                     index=self.getindex(),
                                     bucket_key=None)
        else:
            raise StopIteration


    def iter_next(self):
        """Iterate to next batch.
        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """

        return True


    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """

        return self.data

    def getindex(self):
        """
        Retures
        -------
        index : numpy.array
            The index of current batch
        """
        return np.zeros((self.batch_size,))


    def getpad(self):
        """Get the number of padding examples in current batch.
        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        return 0


class MergeIter(mx.io.DataIter):

    def __init__(self, base_iter, * appending_iter):
        super(MergeIter, self).__init__()

        self.base_iter = base_iter
        self.appending_iter = list(appending_iter)

        self.default_bucket_key = self.base_iter.default_bucket_key
        self.supervised = base_iter.supervised
        self.provide_data = self.base_iter.provide_data[:]
        for iter in self.appending_iter:
            self.provide_data.extend(iter.provide_data)

        self.provide_label = self.base_iter.provide_label

        self.batch_size = base_iter.batch_size

        self.data_names = self.base_iter.get_data_names()[:]

        for iter in self.appending_iter:
            self.data_names.extend(iter.get_data_names())

        self.label_names = self.base_iter.get_label_names()[:]

    def reset(self):
        """Reset the iterator. """
        self.base_iter.reset()
        for iter in self.appending_iter:
            iter.reset()

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)
        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():

            if self.supervised:
                return SupervisedBatch(data=self.getdata(),
                                       data_names=self.get_data_names(),
                                       label=self.getlabel(),
                                       label_names=self.get_label_names(),
                                       pad=self.getpad(),
                                       index=self.getindex(),
                                       bucket_key=self.get_bucket(),
                                       )
            else:
                return UnsupervisedBatch(data=self.getdata(),
                                         data_names=self.get_data_names(),
                                         pad=self.getpad(),
                                         index=self.getindex(),
                                         bucket_key=self.get_bucket())
        else:
            raise StopIteration

    def get_data_names(self):

        return self.data_names


    def get_label_names(self):
        return self.label_names


    def get_bucket(self):

        return self.base_iter.get_bucket()

    def iter_next(self):
        """Iterate to next batch.
        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """

        if not self.base_iter.iter_next():
            return False

        for iter in self.appending_iter:
            if not iter.iter_next():
                return False

        return True


    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """

        data = self.base_iter.getdata()

        for iter in self.appending_iter:
            data.extend(iter.getdata())

        return data

    def getlabel(self):
        """Get label of current batch.
        Returns
        -------
        label : NDArray
            The label of current batch.
        """

        return self.base_iter.getlabel()

    def getindex(self):
        """
        Retures
        -------
        index : numpy.array
            The index of current batch
        """
        return self.base_iter.getindex()

    def getpad(self):
        """Get the number of padding examples in current batch.
        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        return self.base_iter.getpad()