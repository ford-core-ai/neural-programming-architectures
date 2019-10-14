"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.
"""
import numpy as np
import sys
import time

CONFIG = {
    "ENVIRONMENT_LEN": 5,        # 10-element long array for bubble sort task
    "ENVIRONMENT_DEPTH": 11,      # Size of each element vector => One-Hot, Options: 0-9, OOB (10)

    "ARGUMENT_NUM": 3,            # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 11,         # Size of Argument Vector => One-Hot, Options 0-9, Default (10)
    "DEFAULT_ARG_VALUE": 10,      # Default Argument Value

    "PROGRAM_NUM": 10,             # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 5,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 10  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ("PTR", 3, 2),                # Moves Pointer (3 options) either left or right (2 options)
    ("SWAP", 1, 1),               # Swaps PTR1 value (1 option) with PTR2 value (1 option)
    ("BUBBLESORT",),              # Top-Level Bubble Sort Program (calls children routines)
    ("BUBBLE",),                  # Single-pass bubble sort application
    ("BSTEP",),                   # Sort two values and move on
    ("COMPSWAP",),                # Check if values need to be swapped
    ("RSHIFT",),                  # Shifts value Pointers Right (after COMPSWAP)
    ("LSHIFT",),                  # Shifts value Pointers Left (after RESET)
    ("RESET",),                    # Reset pointers to start of array
    ("RETURN",)                   # Return from recursed call
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}


class ScratchPad():           # Addition Environment
    def __init__(self, array, length=CONFIG["ENVIRONMENT_LEN"]):
        # Setup Internal ScratchPad
        self.length = length
        self.scratchpad = np.zeros((self.length), dtype=np.int8)

        # Initialize ScratchPad In1, In2
        self.init_scratchpad(array)

        # Pointers initially all start at the right
        self.val1_ptr, self.val2_ptr, self.iter_ptr = self.ptrs = [0, 1, 1]

    def init_scratchpad(self, array):
        """
        Initialize the scratchpad with the given input numbers (to be added together).
        """
        for i, value in enumerate(array):
            self.scratchpad[i] = value

    def done(self):
        if self.iter_ptr == self.length:
            return True
        else:
            return False

    def swap(self):
        if self[self.val1_ptr] > self[self.val2_ptr]:
            return True
        else:
            return False

    def bstep(self):
        if self.val2_ptr < self.length:
            return True
        else:
            return False

    def lshift(self):
        if 0 <= self.val1_ptr:
            return True
        else:
            return False

    def pretty_print(self):
        print('Array: ', self.scratchpad)
        print('')
        time.sleep(.1)
        sys.stdout.flush()

    def get_env(self):
        env = np.zeros((4,), dtype=np.int32)
        if 0 <= self.val1_ptr < self.length:
            env[1] = 1
        if 0 <= self.val2_ptr < self.length:
            env[2] = 1
        if env[1] == 1 and env[2] == 1:
            if self[self.val1_ptr] <= self[self.val2_ptr]:
                env[0] = 1
        if self.iter_ptr == self.length:
            env[3] = 1
        return env

    def execute(self, prog_id, args):
        if prog_id == 0:               # MOVE!
            ptr, lr = args
            lr = (lr * 2) - 1
            if ptr == 0:
                self.val1_ptr += lr
            elif ptr == 1:
                self.val2_ptr += lr
            elif ptr == 2:
                self.iter_ptr += lr
            else:
                raise NotImplementedError
            self.ptrs = [self.val1_ptr, self.val2_ptr, self.iter_ptr]
        elif prog_id == 1:              # SWAP!
            temp = self[self.val1_ptr]
            self[self.val1_ptr] = self[self.val2_ptr]
            self[self.val2_ptr] = temp

    def __getitem__(self, item):
        if 0 <= item < self.scratchpad.size:
            return self.scratchpad[item]
        else:
            return -1

    def __setitem__(self, key, value):
        self.scratchpad[key] = value


class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)


def get_args(args, arg_in=True):
    if arg_in:
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
    else:
        arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                   range(CONFIG["ARGUMENT_NUM"])]
    if len(args) > 0:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            if i >= len(args):
                arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
            else:
                arg_vec[i][args[i]] = 1
    else:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    return arg_vec.flatten() if arg_in else arg_vec

