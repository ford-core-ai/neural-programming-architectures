"""
train.py

Core training script for the addition task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
from model.npi import NPI
from tasks.bubblesort.bubblesort import BubbleSortCore
from tasks.bubblesort.env.config import CONFIG, get_args, ScratchPad, PROGRAM_SET
import pickle
import tensorflow as tf
import numpy as np

PTR_PID, SWAP_PID = 0, 1
WRITE_OUT, WRITE_CARRY = 0, 1
VAL1_PTR, VAL2_PTR, ITER_PTR = range(3)
LEFT, RIGHT = 0, 1
DATA_PATH = "tasks/bubblesort/data/train.pik"
LOG_PATH = "tasks/bubblesort/log/"


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

        # Start Training
        for ep in range(1, epochs + 1):
            for i in range(len(data)):
                # Setup Environment
                array, steps = data[i]
                scratch = ScratchPad(array)
                x, y = steps[:-1], steps[1:]

                state_stack = []

                # Run through steps, and fit!
                step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
                arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0
                for j in range(len(x)):
                    (prog_name, prog_in_id), arg, term = x[j]
                    (_, prog_out_id), arg_out, term_out = y[j]

                    # Update Environment if MOVE or WRITE
                    if prog_in_id == PTR_PID or prog_in_id == SWAP_PID:
                        scratch.execute(prog_in_id, arg)

                    # Get Environment, Argument Vectors
                    env_in = [scratch.get_env()]
                    arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
                    prog_in, prog_out = [[prog_in_id]], [prog_out_id]
                    term_out = [1] if term_out else [0]

                    temp = len(state_stack)

                    # reset state if we recurse
                    if prog_name == "BUBBLESORT":
                        states = np.zeros([npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])
                        state_stack = [states]
                    elif prog_name == "BSTEP" or prog_name == "LSHIFT":
                        states = np.zeros([npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])
                        state_stack.append(states)
                        # print("state added")
                        # print("stack len: ", len(state_stack))
                    elif prog_name == "RETURN":
                        state_stack.pop()
                        # print("state pop")
                        # print("stack len: ", len(state_stack))

                    if i % 100 == 0:
                        print("prog: |" + "  " * temp, prog_name)
                        # print("env: ", env_in)
                        # print("arg: ", arg_in)
                        # print("state: ", state_stack[-1])

                    # Fit!
                    if prog_out_id == PTR_PID or prog_out_id == SWAP_PID:
                        emb, loss, a, b, c, t_acc, p_acc, a_acc, h_states, _ = sess.run(
                            [npi.program_embedding, npi.arg_loss, npi.terminate, npi.program_distribution, npi.arguments, npi.t_metric, npi.p_metric, npi.a_metrics, npi.h_states, npi.arg_train_op],
                            feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                       npi.y_prog: prog_out, npi.y_term: term_out,
                                       npi.y_args[0]: [arg_out[0]], npi.y_args[1]: [arg_out[1]],
                                       npi.y_args[2]: [arg_out[2]], npi.states: state_stack[-1]})
                        # print("terminate: ", a)
                        # print("prog: ", b)
                        # print("args: ", c)
                        step_arg_loss += loss
                        term_acc += t_acc
                        prog_acc += p_acc
                        arg0_acc += a_acc[0]
                        arg1_acc += a_acc[1]
                        arg2_acc += a_acc[2]
                        num_args += 1
                        state_stack[-1] = np.reshape(h_states, [npi.npi_core_layers, npi.bsz, 2*npi.npi_core_dim])
                    else:
                        emb, loss, a, b, t_acc, p_acc, h_states, _ = sess.run(
                            [npi.program_embedding, npi.default_loss, npi.terminate, npi.program_distribution, npi.t_metric, npi.p_metric, npi.h_states, npi.default_train_op],
                            feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                       npi.y_prog: prog_out, npi.y_term: term_out, npi.states: state_stack[-1]})
                        # print("terminate: ", a)
                        # print("prog: ", b)
                        step_def_loss += loss
                        term_acc += t_acc
                        prog_acc += p_acc
                        state_stack[-1] = np.reshape(h_states, [npi.npi_core_layers, npi.bsz, 2*npi.npi_core_dim])

                    if i % 100 == 0:
                        prog_id = np.argmax(b)
                        prog_name = PROGRAM_SET[prog_id][0]
                        print("guess:|" + "  " * len(state_stack), prog_name)
                        # print(emb)


                print("Epoch {0:02d} Step {1:03d} Default Step Loss {2:05f}, " \
                      "Argument Step Loss {3:05f}, Term: {4:03f}, Prog: {5:03f}, A0: {6:03f}, " \
                      "A1: {7:03f}, A2: {8:03}" \
                      .format(ep, i, step_def_loss / len(x), step_arg_loss / len(x), term_acc / len(x),
                              prog_acc / len(x), arg0_acc / num_args, arg1_acc / num_args,
                              arg2_acc / num_args))

            # Save Model
            saver.save(sess, 'tasks/bubblesort/log/model.ckpt', global_step=len(data)*ep)