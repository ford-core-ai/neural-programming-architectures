"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.bubblesort.bubblesort import BubbleSortCore
from tasks.bubblesort.env.config import CONFIG, get_args, PROGRAM_SET, ScratchPad
import numpy as np
import pickle
import copy
import tensorflow as tf
from tqdm import tqdm

LOG_PATH = "tasks/bubblesort/log/"
CKPT_PATH = "tasks/bubblesort/log/"
TEST_PATH = "tasks/bubblesort/data/test.pik"
PTR_PID, SWAP_PID = 0, 1
W_PTRS = {0: "OUT", 1: "CARRY"}
PTRS = {0: "VAL1_PTR", 1: "VAL2_PTR", 2: "ITER_PTR"}
R_L = {0: "LEFT", 1: "RIGHT"}


def evaluate_bubblesort(verbose=False):
    """
    Load NPI Model from Checkpoint, and initialize REPL, for interactive carry-addition.
    """
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as sess:
        # Load Data
        with open(TEST_PATH, 'rb') as f:
            data = pickle.load(f)

        # Initialize Addition Core
        core = BubbleSortCore()

        # Initialize NPI Model
        npi = NPI(core, CONFIG, LOG_PATH)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(CKPT_PATH)
        saver.restore(sess, ckpt)

        # Run REPL
        repl(sess, npi, data, verbose=verbose)


def repl(session, npi, data, verbose):
    inpt = input('Hit Enter to test, or type anything for more options: ')
    if inpt == "":
        # run through test set
        correct_count = 0
        for array, _ in tqdm(data):
            result = inference(session, npi, array, verbose=verbose)
            correct_count += int(result)

        print("Test Accuracy: %3.2f%%" % (100 * correct_count/len(data)))
    else:
        # run verbose test samples
        while True:
            inpt = input('Enter Two Numbers, or Hit Enter for Random Pair: ')

            if inpt == "":
                array, _ = data[np.random.randint(len(data))]

            else:
                array = map(int, inpt.split())

            inference(session, npi, array, verbose=True)


def inference(session, npi, array, verbose):
    # Separate steps
    if verbose:
        print("")

    # Setup Environment
    scratch = ScratchPad(array)
    prog_name, prog_id, arg, term = 'BUBBLESORT', 2, [], False

    while True:
        # Update Environment if MOVE or WRITE
        if prog_id == PTR_PID or prog_id == SWAP_PID:
            scratch.execute(prog_id, arg)

        # control recursive lstm state
        if prog_name == "BUBBLESORT":
            states = np.zeros([npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])
            state_stack = [states]
        elif prog_name == "BSTEP" or prog_name == "LSHIFT":
            states = np.zeros([npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])
            state_stack.append(states)
        elif prog_name == "RETURN":
            state_stack.pop()

        # Get Environment, Argument Vectors
        env_in, arg_in, prog_in = [scratch.get_env()], [get_args(arg, arg_in=True)], [[prog_id]]
        t, n_p, n_args, h_states = session.run([npi.terminate, npi.program_distribution, npi.arguments, npi.h_states],
                                               feed_dict={npi.env_in: [env_in], npi.arg_in: [arg_in],
                                                          npi.prg_in: prog_in, npi.states: state_stack[-1]})

        if verbose:
            # Print Step Output
            if prog_id == PTR_PID:
                a0, a1 = PTRS[arg[0]], R_L[arg[1]]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            elif prog_id == SWAP_PID:
                a0, a1 = PTRS[arg[0]], PTRS[arg[1]]
                a_str = "[%s, %s]" % (str(a0), str(a1))
            else:
                a_str = "[]"

            # Print Output & Pointers
            print('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term)))
            print('PTR 1: %s, PTR 2: %s, ITER: %s' % (scratch.val1_ptr,
                                                      scratch.val2_ptr,
                                                      scratch.iter_ptr))
            print('VAL 1: %s, VAL 2: %s, DONE: %s' % (scratch[scratch.val1_ptr],
                                                      scratch[scratch.val2_ptr],
                                                      env_in[0][3]))

            # Print Environment
            scratch.pretty_print()

        state_stack[-1] = np.reshape(h_states, [npi.npi_core_layers, npi.bsz, 2 * npi.npi_core_dim])

        if np.argmax(t) == 1:
            # Update Environment if MOVE or WRITE
            if prog_id == PTR_PID or prog_id == SWAP_PID:
                scratch.execute(prog_id, arg)

            # find correct answer
            correct = copy.deepcopy(array)
            correct.sort()

            # compare to system output
            output = scratch.scratchpad
            result = (output == correct).all()

            if verbose:
                print('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(True)))
                print('VAL 1: %s, VAL 2: %s, ITER: %s' % (scratch.val1_ptr,
                                                          scratch.val2_ptr,
                                                          scratch.iter_ptr))

                # Print Environment
                scratch.pretty_print()

                print("Model Output: %s => %s" % (str(array), str(output)))
                print("Correct Out : %s => %s" % (str(array), str(correct)))
                print("Correct!" if result else "Incorrect!")

            break

        else:
            # compute next step inputs
            prog_id = np.argmax(n_p)
            prog_name = PROGRAM_SET[prog_id][0]
            if prog_id == PTR_PID or prog_id == SWAP_PID:
                arg = [np.argmax(n_args[0]), np.argmax(n_args[1])]
            else:
                arg = []
            term = False

    return result