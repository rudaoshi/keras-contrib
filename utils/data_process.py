
import numpy as np

def to_time_distributed_categorical(y, nb_classes=None):

    sample_size, time_steps = y.shape
    if not nb_classes:
        nb_classes = np.max(y)+1

    Y = np.zeros((sample_size, time_steps, nb_classes))

    for i in range(sample_size):
        for j in range(time_steps):
            Y[i, j, y[i, j]] = 1.
    return Y
