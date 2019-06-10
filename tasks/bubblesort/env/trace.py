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

        print("=" * 40)
        # Build Execution Trace
        self.build()
        for trace in self.traces:
            print("~" * 20)
            for step in trace:
                print(step[1], step[-1])

        # Check answer
        self.array.sort()
        true_ans = self.array
        trace_ans = self.scratch.scratchpad
        assert((true_ans == trace_ans).all())

    def construct(self, prog_name, prog_id, args, term):
        # print(prog_id, prog_name, args)
        self.scratch.execute(prog_id, args)
        env = self.scratch.get_env()
        # print(self.scratch.pretty_print())
        self.trace[-1].append([env, prog_name, prog_id, args, term])
        if prog_name in ["BUBBLESORT", "BSTEP", "LSHIFT"]:
            self.trace.append([])
            self.trace[-1].append([env, prog_name, prog_id, args, term])
        elif prog_name == "RETURN":
            # self.trace[-1].append([env, prog_name, prog_id, args, term])
            self.traces.append(self.trace.pop())

        # if len(self.trace) == 0:
        #     self.trace.append([])
        #     self.trace[-1].append([env, prog_name, prog_id, args, term])

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

        self.construct(BSTEP, P[BSTEP], [], False)
        self.bstep()
        self.construct(RETURN, P[RETURN], [], False)

    def bstep(self):
        self.construct(COMPSWAP, P[COMPSWAP], [], False)

        if self.scratch.swap():
            self.construct(SWAP, P[SWAP], [VAL1_PTR, VAL2_PTR], False)

        self.construct(RSHIFT, P[RSHIFT], [], False)
        self.construct(PTR, P[PTR], [VAL1_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [VAL2_PTR, RIGHT], False)

        self.construct(BSTEP, P[BSTEP], [], False)
        if self.scratch.bstep():
            self.bstep()
        self.construct(RETURN, P[RETURN], [], False)

    def reset(self):
        self.construct(RESET, P[RESET], [], False)
        self.construct(LSHIFT, P[LSHIFT], [], False)
        self.lshift()
        self.construct(RETURN, P[RETURN], [], False)

        self.construct(RSHIFT, P[RSHIFT], [], False)
        self.construct(PTR, P[PTR], [VAL1_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [VAL2_PTR, RIGHT], False)
        self.construct(PTR, P[PTR], [ITER_PTR, RIGHT], False)

        # self.scratch.iter_ptr += 1
        if self.scratch.done():
            self.construct(RETURN, P[RETURN], [], True)

            # self.scratch.iter_ptr -= 1
            # self.construct(PTR, P[PTR], [ITER_PTR, RIGHT], True)
        # else:
            # self.scratch.iter_ptr -= 1
            # self.construct(PTR, P[PTR], [ITER_PTR, RIGHT], False)
        # self.construct(RETURN, P[RETURN], [], False)

    def lshift(self):
        # Move Val1 Pointer Left
        self.construct(PTR, P[PTR], [VAL1_PTR, LEFT], False)

        # Move Val2 Pointer Left
        self.construct(PTR, P[PTR], [VAL2_PTR, LEFT], False)

        self.construct(LSHIFT, P[LSHIFT], [], False)

        if self.scratch.lshift():
            self.lshift()
        # exit recursion
        self.construct(RETURN, P[RETURN], [], False)