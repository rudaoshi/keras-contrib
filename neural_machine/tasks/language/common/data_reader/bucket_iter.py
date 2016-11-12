import numpy as np



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
        self.data = [x for x in data]
        self.data_names = data_names
        self.bucket_key = bucket_key
        self.pad = pad
        self.index = index


class SupervisedBatch(object):
    def __init__(self, data, data_names, label, label_names,
                 pad, index,
                 bucket_key):
        #logging.log(logging.DEBUG, "Data:"  + " ".join([str(x.shape) for x in data ]))
        #logging.log(logging.DEBUG, "Label:" + " ".join([str(x.shape) for x in label]))

        self.data = [x for x in data]
        self.label = [x for x in label]
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


class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        self.batch_size = 0

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator. """
        pass

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
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Iterate to next batch.

        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        pass

    def getlabel(self):
        """Get label of current batch.

        Returns
        -------
        label : NDArray
            The label of current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The index of current batch
        """
        return None

    def getpad(self):
        """Get the number of padding examples in current batch.

        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        pass



class DummyIter(DataIter):
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

class DataBatch(object):
    """Default object for holding a mini-batch of data and related information."""
    def __init__(self, data, label, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        # the following properties are only used when bucketing is used
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

def pad(l, bucket_size):

    data = []

    for j in range(len(bucket_size)):
        np_array = np.zeros((len(l[j]), bucket_size[j]))
        for i in range(len(l[j])):
            np_array[i, :len(l[j][i])] = l[j][i]

        data.append(np_array)
    return data

def bucket_contains(bucket_a, bucket_b):
    return min([bucket_a[j] - bucket_b[j] for j in range(len(bucket_b))]) >= 0

def bucket_distance(bucket_a, bucket_b):
    return sum([bucket_a[j] - bucket_b[j] for j in range(len(bucket_b))])

import logging
from collections import defaultdict
class BucketIter(DataIter):


    def gen_buckets(self, batch_size, max_pad_num):
        shape_cap_map = Counter()

        for sample in self.problem.samples():
            shape = self.sample_shape(sample)
            shape_cap_map[shape] += 1

        #shape_capacity = sorted(shape_cap_map.iteritems(), key=lambda x: sum(x[0]))
        max_bucket = tuple(np.max(np.array(shape_cap_map.keys()),axis=0))

        bucket_shape_map = defaultdict(set)
        bucket_cap_map = Counter()

        for shape, cap in shape_cap_map.iteritems():

            target_bucket = tuple([shape[i] if shape[i]%max_pad_num == 0 else (shape[i]/max_pad_num + 1) * max_pad_num
                                   for i in range(len(shape))])
            bucket_shape_map[target_bucket].add(shape)
            bucket_cap_map[target_bucket] += cap

        while True:

            minor_buckets = [bucket for bucket, cap in bucket_cap_map.iteritems() if cap < batch_size]
            update = False
            for minor_bucket in minor_buckets:

                if bucket_cap_map[minor_bucket] >= batch_size:
                    continue

                neighbors = sorted([x for x in bucket_cap_map.keys() if bucket_contains(x, minor_bucket) and x != minor_bucket],
                                   key=lambda x: bucket_distance(x, minor_bucket))
                if neighbors:

                    logging.info("{0} with {1} sample been merged.".format(minor_bucket, bucket_cap_map[minor_bucket]))

                    merge_target = neighbors[0]
                    bucket_cap_map[merge_target] += bucket_cap_map[minor_bucket]
                    del bucket_cap_map[minor_bucket]

                    bucket_shape_map[merge_target].update(bucket_shape_map[minor_bucket])
                    del bucket_shape_map[minor_bucket]

                    update = True

            if not update:

                logging.info("No merge can be found")

                if minor_buckets:
                    cur_max_bucket = tuple(np.max(np.array(minor_buckets), axis=0))
                    total_num = 0
                    for minor_bucket in minor_buckets:
                        total_num += bucket_cap_map[minor_bucket]
                        bucket_cap_map[cur_max_bucket] += bucket_cap_map[minor_bucket]
                        del bucket_cap_map[minor_bucket]

                        bucket_shape_map[cur_max_bucket].update(bucket_shape_map[minor_bucket])
                        del bucket_shape_map[minor_bucket]

                    logging.info("{0} sample been merged to biggest bucket {1}.".format(total_num, cur_max_bucket))

                break

        buckets = bucket_shape_map.keys()
        bucket_id_map = dict((x, id) for id, x in enumerate(buckets))

        shape_bucket_id_map = dict()
        for bucket, shapes in bucket_shape_map.iteritems():

            for shape in shapes:
                shape_bucket_id_map[shape] = bucket_id_map[bucket]

        for shape in shape_cap_map:

            target_bucket_id = shape_bucket_id_map[shape]
            target_bucket = buckets[target_bucket_id]

            assert min([target_bucket[j] - shape[j] for j in range(len(shape))]) > 0


        logging.info("{0} buckets with max capacity {1}".format(len(buckets), max_bucket))

        return buckets, max_bucket, shape_bucket_id_map

    def sample_shape(self, sample):

        if self.supervised:
            shape = [len(x) for x in sample[0]] + [len(x) for x in sample[1]]
        else:
            shape = [len(x) for x in sample]

        return tuple(shape)

    def __init__(self, problem, batch_size, max_pad_num = 10, batch_pad = False):
        super(BucketIter, self).__init__()

        self.batch_pad = batch_pad

        self.problem = problem
        self.supervised = self.problem.is_supervised()

        self.data_names = problem.data_names()
        self.label_names = problem.label_names()
        self.buckets, self.default_bucket_key, bucket_map = self.gen_buckets(batch_size, max_pad_num)

        self.data = [[[] for _ in range(len(self.data_names))]
                     for _ in range(len(self.buckets))]

        if self.supervised:
            self.label = [[[] for _ in range(len(self.data_names))]
                          for _ in range(len(self.buckets))]

        for n, sample in enumerate(self.problem.samples()):

            shape = self.sample_shape(sample)
            idx = bucket_map[shape]

            if self.supervised:
                for j in range(len(self.data_names)):
                    self.data[idx][j].append(sample[0][j])
                for j in range(len(self.label_names)):
                    self.label[idx][j].append(sample[1][j])
            else:
                for j in range(len(self.data_names)):
                    self.data[idx][j].append(sample[j])

            #if n % 1000 == 0:
            #    logging.debug("{0} samples imported into bucket".format(n))

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

        if (j+1) * self.batch_size >= self.data[self.shuffled_bucket_idx[i]][0].shape[0]:
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

            if self.batch_pad:
                return [np.vstack((x,
                             np.zeros((self.batch_size-x.shape[0],
                                       x.shape[1])))) for x in cur_data]
            else:
                return cur_data

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
            if self.batch_pad:
                return [np.vstack((x,
                             np.zeros((self.batch_size-x.shape[0],
                                       x.shape[1])))) for x in cur_label]
            else:
                return cur_label

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
            if self.batch_pad:
                return self.batch_size - len(self.label[bucket_id])% self.batch_size
            else:
                return 0


class RepeatedAppendIter(DataIter):

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


class MergeIter(DataIter):

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
