"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
from model.npi import NPI
from tasks.bubblesort.bubblesort import BubbleSortCore
from tasks.bubblesort.env.config import CONFIG, get_args
import pickle
import tensorflow as tf
import numpy as np

PTR_PID, SWAP_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
VAL1_PTR, VAL2_PTR, ITER_PTR = range(3)
LEFT, RIGHT = 0, 1
DATA_PATH = "tasks/bubblesort/data/train.pik"
LOG_PATH = "tasks/bubblesort/log/"
PRINT_EVERY = 100

def train_bubblesort(epochs, verbose=0):
    """
    Instantiates an Addition Core, NPI, then loads and fits model to data.

    :param epochs: Number of epochs to train for.
    """
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load Data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Initialize Addition Core
    print('Initializing Bubble Sort Core!')
    core = BubbleSortCore()

    # Initialize NPI Model
    print('Initializing NPI Model!')
    npi = NPI(core, CONFIG, LOG_PATH, verbose=verbose)

    num_params = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
    print('Total number of trainable parameters: ', int(num_params))

    # Initialize TF Saver
    saver = tf.train.Saver()

    # Initialize TF Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step_arg_loss, term_acc, prog_acc = 0.0, 0.0, 0.0
        arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0.0
        i = 0

        # Start Training
        for ep in range(1, epochs + 1):
            for _, full_trace in data:
                for steps in full_trace:
                    # Setup Environment
                    x, y = steps[:-1], steps[1:]

                    env_in, prog_name, prog_in_id, args_in, term = list(map(list, zip(*x)))
                    _, prog_name_out, prog_out_id, args_out, term_out = list(map(list, zip(*y)))

                    env_in = np.array(env_in)
                    env_in = np.expand_dims(env_in, axis=0)

                    args_in = np.array([get_args(arg, arg_in=True) for arg in args_in])
                    args_out = np.array([get_args(arg, arg_in=False) for arg in args_out])
                    args_out = np.transpose(args_out, (1, 0, 2))

                    prog_in, prog_out = [prog_in_id], [prog_out_id]
                    prog_out = np.array(prog_out)

                    term_out = np.array(term_out)
                    term_out.astype(int)
                    term_out = np.expand_dims(term_out, axis=0)

                    states = np.zeros([npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])

                    # Fit!
                    loss, t_acc, p_acc, a_acc, _, a = sess.run(
                        [npi.arg_loss, npi.t_metric, npi.p_metric, npi.a_metrics, npi.arg_train_op, npi.core.program_key],
                        feed_dict={npi.env_in: env_in, npi.arg_in: [args_in], npi.prg_in: prog_in,
                                   npi.y_prog: prog_out, npi.y_term: term_out,
                                   npi.y_args[0]: [args_out[0, :, :]], npi.y_args[1]: [args_out[1, :, :]],
                                   npi.y_args[2]: [args_out[2, :, :]], npi.states: states})
                    step_arg_loss += loss
                    term_acc += t_acc/PRINT_EVERY
                    prog_acc += p_acc/PRINT_EVERY
                    arg0_acc += a_acc[0]/PRINT_EVERY
                    arg1_acc += a_acc[1]/PRINT_EVERY
                    arg2_acc += a_acc[2]/PRINT_EVERY
                    num_args += len(x)

                    if i % PRINT_EVERY == 0:
                        print("Epoch {0:02d} Step {1:03d} " \
                              "Argument Step Loss {2:05f}, Term: {3:03f}, Prog: {4:03f}, A0: {5:03f}, " \
                              "A1: {6:03f}, A2: {7:03f}" \
                              .format(ep, i, step_arg_loss / num_args, term_acc,
                                      prog_acc, arg0_acc, arg1_acc, arg2_acc))
                        step_arg_loss, term_acc, prog_acc = 0.0, 0.0, 0.0
                        arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0.0
                    i += 1

            # Save Model
            saver.save(sess, 'tasks/bubblesort/log/model.ckpt', global_step=len(data)*ep)