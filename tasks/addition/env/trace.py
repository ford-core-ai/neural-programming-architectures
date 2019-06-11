"""
trace.py

Core class definition for a trace object => given a pair of integers to add, builds the execution
trace, calling the specified subprograms.
"""
from tasks.addition.env.config import ScratchPad, PROGRAM_ID as P
ADD, ADD1, WRITE, LSHIFT, CARRY, MOVE_PTR, RETURN = "ADD", "ADD1", "WRITE", "LSHIFT", "CARRY", "MOVE_PTR", "RETURN"
WRITE_OUT, WRITE_CARRY = 0, 1
IN1_PTR, IN2_PTR, CARRY_PTR, OUT_PTR = range(4)
LEFT, RIGHT = 0, 1


class Trace():
    def __init__(self, in1, in2, debug=False):
        """
        Instantiates a trace object, and builds the exact execution pipeline for adding the given
        parameters.
        """
        self.in1, self.in2, self.debug = in1, in2, debug
        self.trace, self.scratch = [[]], ScratchPad(in1, in2)
        self.traces = []

        print("=" * 40)
        # Build Execution Trace
        self.build()
        for trace in self.traces:
            for step in trace:
                print(step[1])

        # Check answer
        true_ans = self.in1 + self.in2
        trace_ans = int("".join(map(str, map(int, self.scratch[3]))))

        assert(true_ans == trace_ans)

    def construct(self, prog_name, prog_id, args, term):
        # execute the provided program in the scratchpad
        self.scratch.execute(prog_id, args)
        # get environment after the program has been executed
        env = self.scratch.get_env()
        # add all input/output terms to trace
        self.trace[-1].append([env, prog_name, prog_id, args, term])
        # manually control recursion for specific programs
        if prog_name == "ADD":
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
            self.add1()
            self.lshift()
        for _ in self.trace:
            self.construct(RETURN, P[RETURN], [], False)

    def add1(self):
        # Recurse into Add Subroutine
        self.construct(ADD, P[ADD], [], False)
        self.construct(ADD1, P[ADD1], [], False)
        out, carry = self.scratch.add1()

        # Write to Output
        self.construct(WRITE, P[WRITE], [WRITE_OUT, out], False)

        # Carry Condition
        if carry > 0:
            self.carry(carry)

    def carry(self, carry_val):
        # Call Carry Subroutine
        self.construct(CARRY, P[CARRY], [], False)

        # Shift Carry Pointer Left
        self.construct(MOVE_PTR, P[MOVE_PTR], [CARRY_PTR, LEFT], False)

        # Write Carry Value
        self.construct(WRITE, P[WRITE], [WRITE_CARRY, carry_val], False)

        # Shift Carry Pointer Right
        self.construct(MOVE_PTR, P[MOVE_PTR], [CARRY_PTR, RIGHT], False)

    def lshift(self):

        # Move Inp1 Pointer Left
        self.construct(LSHIFT, P[LSHIFT], [], False)

        # Move Inp1 Pointer Left
        self.construct(MOVE_PTR, P[MOVE_PTR], [IN1_PTR, LEFT], False)

        # Move Inp2 Pointer Left
        self.construct(MOVE_PTR, P[MOVE_PTR], [IN2_PTR, LEFT], False)

        # Move Carry Pointer Left
        self.construct(MOVE_PTR, P[MOVE_PTR], [CARRY_PTR, LEFT], False)

        # Move Out Pointer Left
        self.construct(MOVE_PTR, P[MOVE_PTR], [OUT_PTR, LEFT], False)

        # check if done
        if self.scratch.done():
            self.construct(RETURN, P[RETURN], [], True)