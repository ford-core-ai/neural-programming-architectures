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
        self.trace, self.scratch = [[]], ScratchPad(array)
        self.traces = []

        # Build Execution Trace
        self.build()

        # Check answer
        self.array.sort()
        true_ans = self.array
        trace_ans = self.scratch.scratchpad
        assert((true_ans == trace_ans).all())

    def construct(self, prog_name, prog_id, args, term):
        # execute the provided program in the scratchpad
        self.scratch.execute(prog_id, args)
        # get environment after the program has been executed
        env = self.scratch.get_env()
        # add all input/output terms to trace
        self.trace[-1].append([env, prog_name, prog_id, args, term])
        # manually control recursion for specific programs
        if prog_name in ["BUBBLESORT", "BSTEP", "LSHIFT"]:
            self.trace.append([])
            self.trace[-1].append([env, prog_name, prog_id, args, term])
        elif prog_name == "RETURN":
            self.traces.append(self.trace.pop())

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
        for _ in self.trace:
            self.construct(RETURN, P[RETURN], [], False)

    def bubble(self):
        # Recurse into Bubble Subroutine
        self.construct(BUBBLESORT, P[BUBBLESORT], [], False)
        self.construct(BUBBLE, P[BUBBLE], [], False)

        # call recursive BSTEP
        self.construct(BSTEP, P[BSTEP], [], True)
        self.bstep()
        # self.construct(RETURN, P[RETURN], [], False)

    def bstep(self):
        self.construct(COMPSWAP, P[COMPSWAP], [], False)
        # optionally swap digits
        if self.scratch.swap():
            self.construct(SWAP, P[SWAP], [VAL1_PTR, VAL2_PTR], False)

        # move ptrs to next array indices
        self.construct(RSHIFT, P[RSHIFT], [], False)
        self.construct(PTR, P[PTR], [VAL1_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [VAL2_PTR, RIGHT], False)

        # optionally recurse
        self.construct(BSTEP, P[BSTEP], [], True)
        if self.scratch.bstep():
            self.bstep()
        # self.construct(RETURN, P[RETURN], [], False)

    def reset(self):
        self.construct(RESET, P[RESET], [], False)
        # recursively move ptrs back to the start of the array
        self.construct(LSHIFT, P[LSHIFT], [], False)
        self.lshift()
        self.construct(RETURN, P[RETURN], [], False)

        # shift all pointers for next iteration
        self.construct(RSHIFT, P[RSHIFT], [], False)
        self.construct(PTR, P[PTR], [VAL1_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [VAL2_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [ITER_PTR, RIGHT], False)

        # set termination flag if algorithm complete
        if self.scratch.done():
            self.construct(RETURN, P[RETURN], [], True)

    def lshift(self):
        # Move Val1 Pointer Left
        self.construct(PTR, P[PTR], [VAL1_PTR, LEFT], False)

        # Move Val2 Pointer Left
        self.construct(PTR, P[PTR], [VAL2_PTR, LEFT], False)

        # optionally recurse
        self.construct(LSHIFT, P[LSHIFT], [], True)
        if self.scratch.lshift():
            self.lshift()
        # self.construct(RETURN, P[RETURN], [], False)