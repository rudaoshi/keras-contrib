
import numpy as np

def to_time_distributed_categorical(y, nb_classes=None):

    if isinstance(y,  np.ndarray):  # sequence with same length
        sample_size, time_steps = y.shape
        if not nb_classes:
            nb_classes = np.max(y)+1

        Y = np.zeros((sample_size, time_steps, nb_classes))

        for i in range(sample_size):
            for j in range(time_steps):
                Y[i, j, y[i, j]] = 1.
        return Y
    elif isinstance(y, list):   # sequence with variable length

        Y = []

        for seq in y:
            time_steps = len(seq)
            cur_Y = np.zeros((time_steps, nb_classes))
            for j in range(time_steps):
                cur_Y[j, seq[j]] = 1.

            Y.append(cur_Y)

        return Y
