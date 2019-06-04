"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
import pickle

import numpy as np

from tasks.bubblesort.env.trace import Trace


def generate_bubblesort(prefix, num_examples, debug=False, maximum=10000000000, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    data = []
    for i in range(num_examples):
        array = np.random.randint(10, size=5)
        if debug and i % debug_every == 0:
            trace = Trace(array, True).trace
        else:
            trace = Trace(array).trace
        data.append((array, trace))

    # print(data)
    with open('tasks/bubblesort/data/{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)