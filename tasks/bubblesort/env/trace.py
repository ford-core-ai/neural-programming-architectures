"""
trace.py

Core class definition for a trace object => given a pair of integers to add, builds the execution
trace, calling the specified subprograms.
"""
import copy

from tasks.bubblesort.env.config import ScratchPad, PROGRAM_ID as P
PTR, SWAP, BUBBLESORT, BUBBLE, BSTEP = "PTR", "SWAP", "BUBBLESORT", "BUBBLE", "BSTEP"
COMPSWAP, RSHIFT, LSHIFT, RESET, RETURN = "COMPSWAP", "RSHIFT", "LSHIFT", "RESET", "RETURN"
WRITE_OUT, WRITE_CARRY = 0, 1
VAL1_PTR, VAL2_PTR, ITER_PTR = range(3)
LEFT, RIGHT = 0, 1


class Trace():
    def __init__(self, array, debug=False):
        """
        Instantiates a trace object, and builds the exact execution pipeline for adding the given
        parameters.
        """
        self.array, self.debug = copy.deepcopy(array), debug
        self.trace, self.scratch = [], ScratchPad(array)

        # Build Execution Trace
        self.build()

        # Check answer
        self.array.sort()
        true_ans = self.array
        trace_ans = self.scratch.scratchpad
        assert((true_ans == trace_ans).all())

    def build(self):
        """
        Builds execution trace, adding individual steps to the instance variable trace. Each
        step is represented by a triple (program_id : Integer, args : List, terminate: Boolean). If
        a subroutine doesn't take arguments, the empty list is returned.
        """
        # Execute Trace
        while not self.scratch.done():
            self.bubble()
            self.reset()

    def bubble(self):
        # Recurse into Bubble Subroutine
        self.trace.append(((BUBBLESORT, P[BUBBLESORT]), [], False))
        self.trace.append(((BUBBLE, P[BUBBLE]), [], False))
        # self.scratch.prep()

        self.trace.append(((BSTEP, P[BSTEP]), [], False))
        self.bstep()
        self.trace.append(((RETURN, P[RETURN]), [], False))

    def bstep(self):
        self.trace.append(((COMPSWAP, P[COMPSWAP]), [], False))
        swap = self.scratch.compswap(debug=self.debug)

        if swap:
            self.trace.append(((SWAP, P[SWAP]), [VAL1_PTR, VAL2_PTR], False))

        self.trace.append(((RSHIFT, P[RSHIFT]), [], False))
        self.trace.append(((PTR, P[PTR]), [VAL1_PTR, RIGHT], False))
        self.trace.append(((PTR, P[PTR]), [VAL2_PTR, RIGHT], False))
        self.trace.append(((BSTEP, P[BSTEP]), [], False))
        ptr2, length = self.scratch.rshift()
        if ptr2 < length:
            self.bstep()
        self.trace.append(((RETURN, P[RETURN]), [], False))

    # def reset(self):
    #     self.trace.append(((RESET, P[RESET]), [], False))
    #     self.trace.append(((LSHIFT, P[LSHIFT]), [], False))
    #     steps = self.scratch.reset()
    #
    #     for _ in range(steps):
    #         self.lshift()
    #
    #     for _ in range(steps):
    #         self.trace.append(((RETURN, P[RETURN]), [], False))
    #
    #     self.trace.append(((RETURN, P[RETURN]), [], False))
    #
    #     if self.scratch.done():
    #         self.trace.append(((PTR, P[PTR]), [ITER_PTR, RIGHT], True))
    #     else:
    #         self.trace.append(((PTR, P[PTR]), [ITER_PTR, RIGHT], False))

    def reset(self):
        self.trace.append(((RESET, P[RESET]), [], False))
        self.trace.append(((LSHIFT, P[LSHIFT]), [], False))
        self.lshift()
        self.trace.append(((RETURN, P[RETURN]), [], False))

        self.trace.append(((RSHIFT, P[RSHIFT]), [], False))
        self.trace.append(((PTR, P[PTR]), [VAL1_PTR, RIGHT], False))
        self.trace.append(((PTR, P[PTR]), [VAL2_PTR, RIGHT], False))
        self.scratch.rshift()
        self.scratch.iter_ptr += 1

        if self.scratch.done():
            self.trace.append(((PTR, P[PTR]), [ITER_PTR, RIGHT], True))
        else:
            self.trace.append(((PTR, P[PTR]), [ITER_PTR, RIGHT], False))

    def lshift(self):
        # Move Val1 Pointer Left
        self.trace.append(((PTR, P[PTR]), [VAL1_PTR, LEFT], False))

        # Move Val2 Pointer Left
        self.trace.append(((PTR, P[PTR]), [VAL2_PTR, LEFT], False))

        self.trace.append(((LSHIFT, P[LSHIFT]), [], False))

        ptr1 = self.scratch.lshift()
        if 0 <= ptr1:
            self.lshift()
        self.trace.append(((RETURN, P[RETURN]), [], False))