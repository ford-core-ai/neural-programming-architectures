# Neural Programming Architecture

Neural Programming Architecture implementation, in Tensorflow (V1.13). Based on the original
[Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/abs/1704.06611) paper, by Cai, Shin , & Song.

This implementation is an altered version of [Ford Core AI's Neural Programmer-Interpreter repo](https://github.com/ford-core-ai/neural-programmer-interpreter), which was based on [Sidd Karamcheti's Neural Programmer-Interpreter repo](https://github.com/siddk/npi).

The discussion below covers technical details of the NPI architecture, which is largely copied to implement NPA.
The only addition made is the network's ability to call itself, and tracking of the network state throughout the recursion layers.

## NPI Overview ##

A Neural Programmer-Interpreter can be decomposed into the following components (each of which are
implemented either in `npi.py`, or `[task-name].py`:

+ **Neural Programmer-Interpreter Core**: Simple LSTM Network (**f_lstm**), with hidden states *h_t*, *c_t*
    - Behaves like a traffic controller ==> Based on task-specific encoder inputs, serves as a 
      router, outputting high-dimensional information that encodes information about which 
      subroutine to call next, and with what arguments.
    - Shared across all tasks ==> only shared, central component.

+ **Task-Specific Encoder Network**: Architecture depends on specific task, but can also be trained 
                                 via gradient descent (**f_enc**). 
    - Given the task environment, and previous subroutine arguments, generates a fixed-length 
      state encoding s_t

+ **Program Termination Network**: Feed-Forward Network (**f_end**), takes LSTM Controller hidden state 
                               *h_t* and outputs a probability of terminating execution.
    - Paper uses a threshold value of 0.5 to determine whether to stop, and terminate, or 
      call next subroutine.

+ **Subroutine Lookup Network**: Feed-Forward Network (**f_prog**), takes LSTM Controller hidden state 
                             *h_t* and outputs a key embedding *k_t* to look up next subroutine to 
                             be called.
    - Subroutine information is stored in two matrices, M_key (N x K), and M_prog (N x P), where 
      each of the N rows denotes a different subroutine, and where K and P are the 
      dimensionality of the key and program embeddings respectively. 
    - The next subroutine is chosen as follows:
        + i* = argmax_i (M_i_key)^T k_t ==> Cosine similarity between the predicted key embedding 
                                           and each of the program keys in M_key
        + p_(t + 1) = M_i*_prog
    - Note: To generate probabilities of next program, for training the network, I took a
            softmax over the array of cosine similarities, and used those as my next-program
            probabilities.

+ **Argument Networks**: Feed-Forward Networks (**f_arg**), takes LSTM Controller hidden state h_t and 
                     outputs subroutine arguments a_(t + 1).
    - Note: In this implementation, I built separate networks for each of the three arguments.
        
### Directory Structure ###
    + model/
        - npi.py: Core model definition for the Neural-Programmer Interpreter. Builds the shared
        NPI LSTM Controller, the Termination Network, the Program ID Network, as well as the 
        specific Argument Networks.
        
    + tasks/
        - [task-name]/
            + data/ - Contains training/test execution traces, stored in Python serialized (.pik)
                      format. Each .pik file contains a list of examples, where each example is 
                      stored as a triple (in1, in2, trace), where in1/in2 are the numbers to be 
                      added, and trace contains the specific execution trace.
            + env/ - Contains the environment/task-specific code, including any Task Configuration
                     parameters, trace-building helpers, etc.
            + [task-name].py - Contains model definition code for the Task-Specific Core - contains
                               task-specific TF placeholders, as well as the [f_enc] encoder 
                               environment-encoder network.
            + train.py - Task-Specific Training Script - implements train_task() function.
            + eval.py - Task-Specific Evaluation Script - implements evaluate_task() function.
    
    + main.py - Core runner, accepts TF Flags for generating data (with specified number of 
                examples), training, saving, and evaluating model.
                
### Training Procedure ###

To train the NPA model for the addition-task, call `python main.py --generate --do_train` from the root of this directory.
This will generate the necessary train and test trace data for the addition-task, and start training.
Once training is complete, the system will enter evaluation mode (outlined below).
                
                
### Evaluation Procedure ###

To test addition-task NPA code interactively, just call `python main.py` from the root of this
directory, provided that you have saved model checkpoints in `tasks/addition/log/`. 
This will drop you into a REPL, where you can enter two numbers to be added, and step through the
predicted execution trace. 

Note that for the time being, numbers much be smaller than 1000000000. This is not because of any 
limitations on the part of the NPA, but because of the backend helper functions that display the 
trace.

A sample execution trace for the addition `18 + 7 = 25` can be found below. Note that this is trace
is produced entirely by the NPA => There is are no external guides provided except for the initial
call to "ADD 18 7".

It should be noted that this sample differs from [Sidd Karamcheti's Neural Programmer-Interpreter repo](https://github.com/siddk/npi) in several key ways.
First, explicit commands to the LSHIFT function were added to more accurately reflect the NPI paper.
Much more crucially, you'll notice that the program continues adding the leading zeros until the end of the scratch pad.
This is to avoid failure cases in which the network pointers all show 0, but the addition is not yet complete, like 1007 + 3.
The original network would incorrectly return 1007 + 3 = 10, because it was taught to stop early if the pointers all showed zeros.

```
    Enter Two Numbers, or Hit Enter for Random Pair: 18 7
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000000
    -----------------------
    Output :     0000000000
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000000
    -----------------------
    Output :     0000000000
    
    Step: WRITE, Arguments: [OUT, 5], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000000
    -----------------------
    Output :     0000000005
    
    Step: CARRY, Arguments: [], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000000
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -2, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000000
    -----------------------
    Output :     0000000005
    
    Step: WRITE, Arguments: [CARRY, 1], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -2, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, RIGHT], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -1, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -2, IN 2: -1, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -1, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -1
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000005
    
    Step: WRITE, Arguments: [OUT, 2], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -2, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -3, IN 2: -2, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -2, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -2
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -3, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -4, IN 2: -3, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -3, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -3
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -4, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -5, IN 2: -4, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -4, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -4
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -5, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -6, IN 2: -5, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -5, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -5
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -6, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -7, IN 2: -6, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -6, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -6
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -7, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -8, IN 2: -7, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -7, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -7
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -8, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -9, IN 2: -8, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -8, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -8
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -9, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -10, IN 2: -9, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -9, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -9
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [OUT_PTR, LEFT], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD, Arguments: [], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: ADD1, Arguments: [], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: WRITE, Arguments: [OUT, 0], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: LSHIFT, Arguments: [], Terminate: False
    IN 1: -10, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN1_PTR, LEFT], Terminate: False
    IN 1: -11, IN 2: -10, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [IN2_PTR, LEFT], Terminate: False
    IN 1: -11, IN 2: -11, CARRY: -10, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: False
    IN 1: -11, IN 2: -11, CARRY: -11, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Step: MOVE_PTR, Arguments: [CARRY_PTR, LEFT], Terminate: True
    IN 1: -11, IN 2: -11, CARRY: -12, OUT: -10
    Input 1:     0000000018
    Input 2:     0000000007
    Carry  :     0000000010
    -----------------------
    Output :     0000000025
    
    Model Output: 18 + 7 = 25
    Correct Out : 18 + 7 = 25
    Correct!
```