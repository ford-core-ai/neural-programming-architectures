"""
npi.py
Core model definition script for the Neural Programmer-Interpreter.
"""
import tensorflow as tf


class NPI():
    def __init__(self, core, config, log_path, npi_core_dim=256, npi_core_layers=2, verbose=0):
        """
        Instantiate an NPI Model, with the necessary hyperparameters, including the task-specific
        core.

        :param core: Task-Specific Core, with fields representing the environment state vector,
                     the input placeholders, and the program embedding.
        :param config: Task-Specific Configuration Dictionary, with fields representing the
                       necessary parameters.
        :param log_path: Path to save network checkpoint and tensorboard log files.
        """
        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.arg_in, self.prg_in = core.env_in, core.arg_in, core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding
        self.num_args, self.arg_depth = config["ARGUMENT_NUM"], config["ARGUMENT_DEPTH"]
        self.num_progs, self.key_dim = config["PROGRAM_NUM"], config["PROGRAM_KEY_SIZE"]
        self.log_path, self.verbose = log_path, verbose

        # Setup LSTM State Control
        self.lstm_stateful = True

        # Setup Label Placeholders
        self.y_term = tf.placeholder(tf.int64, shape=[None], name='Termination_Y')
        self.y_prog = tf.placeholder(tf.int64, shape=[None], name='Program_Y')
        self.y_args = [tf.placeholder(tf.int64, shape=[None, self.arg_depth],
                                      name='Arg{}_Y'.format(str(i))) for i in range(self.num_args)]

        # Build NPI LSTM Core, hidden state
        self.states = tf.placeholder(tf.float32, shape=[self.npi_core_layers, self.bsz, 2*self.npi_core_dim],
                                     name='LSTM_States')
        self.h_states = []
        self.split_states = tf.split(self.states, self.npi_core_layers, axis=0)
        for split_state in self.split_states:
            self.h_states.append(tf.split(tf.squeeze(split_state, axis=0), 2, axis=-1))

        self.h = self.npi_core()

        # Build Termination Network => Returns probability of terminating
        self.terminate = self.terminate_net()

        # Build Key Network => Generates probability distribution over programs
        self.program_distribution = self.key_net()

        # Build Argument Networks => Generates list of argument distributions
        self.arguments = self.argument_net()

        # Build Losses
        self.t_loss, self.p_loss, self.a_losses = self.build_losses()
        self.default_loss = 2 * self.t_loss + self.p_loss
        self.arg_loss = 0.25 * sum([self.t_loss, self.p_loss]) + sum(self.a_losses)

        # Build Optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, 10000, 0.95,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Build Metrics
        self.t_metric, self.p_metric, self.a_metrics = self.build_metrics()
        self.metrics = [self.t_metric, self.p_metric] + self.a_metrics

        # Build Train Ops
        self.default_train_op = self.opt.minimize(self.default_loss, global_step=self.global_step)
        self.arg_train_op = self.opt.minimize(self.arg_loss, global_step=self.global_step)

    # def reset_state(self):
    #     """
    #     Zero NPI Core LSTM Hidden States. LSTM States are represented as a Tuple, consisting of the
    #     LSTM C State, and the LSTM H State (in that order: (c, h)).
    #     """
    #     zero_state = [tf.zeros([self.bsz, self.npi_core_dim]), tf.zeros([self.bsz, self.npi_core_dim])]
    #     self.h_states = [zero_state for _ in range(self.npi_core_layers)]
    #     self.h_states[0][0] = tf.Print(self.h_states[0][0], [self.h_states], message="states reset:")
    #     print(self.h_states)

    def npi_core(self):
        """
        Build the NPI LSTM core, feeding the program embedding and state encoding to a multi-layered
        LSTM, returning the h-state of the final LSTM layer.

        References: Reed, de Freitas [2]
        """
        s_in = self.state_encoding                               # Shape: [bsz, state_dim]
        p_in = self.program_embedding                            # Shape: [bsz, 1, program_dim]

        # Reshape state_in
        s_in = tf.expand_dims(s_in, axis=1)    # Shape: [bsz, 1, state_dim]

        # Concatenate s_in, p_in
        c = tf.concat([s_in, p_in], axis=2)        # Shape: [bsz, 1, state + prog]

        # Feed through Multi-Layer LSTM
        for i in range(self.npi_core_layers):
            # c = tf.Print(c, [self.h_states[i]], message="before: ")
            rnn = tf.keras.layers.LSTMCell(self.npi_core_dim)
            c, self.h_states[i] = rnn(c, self.h_states[i])
            # c = tf.Print(c, [self.h_states[i]], message="after: ")

        # Return Top-Most LSTM H-State
        top_state = tf.split(c, [-1, 1], axis=1)[1]
        top_state = tf.squeeze(top_state, axis=1)
        return top_state                                         # Shape: [bsz, npi_core_dim]

    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.

        References: Reed, de Freitas [3]
        """
        p_terminate = tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2,
                                            bias_regularizer=tf.keras.regularizers.l2)(self.h)
        return p_terminate                                      # Shape: [bsz, 2]

    def key_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.

        References: Reed, de Freitas [3, 4]
        """
        # Get Key from Key Network
        hidden = tf.keras.layers.Dense(self.key_dim, activation=tf.keras.activations.elu,
                                       kernel_regularizer=tf.keras.regularizers.l2,
                                       bias_regularizer=tf.keras.regularizers.l2)(self.h)
        key = tf.keras.layers.Dense(self.key_dim,
                                    kernel_initializer=tf.truncated_normal_initializer)(hidden)  # Shape: [bsz, key_dim]

        # Perform dot product operation, then softmax over all options to generate distribution
        key = tf.reshape(key, [-1, 1, self.key_dim])
        key = tf.tile(key, [1, self.num_progs, 1])             # Shape: [bsz, n_progs, key_dim]
        prog_sim = tf.multiply(key, self.core.program_key)          # Shape: [bsz, n_progs, key_dim]
        prog_dist = tf.reduce_sum(prog_sim, [2])               # Shape: [bsz, n_progs]
        return prog_dist

    def argument_net(self):
        """
        Build the NPI Argument Networks (a separate net for each argument), each of which takes in
        the NPI Core Hidden State, and returns a softmax over the argument dimension.

        References: Reed, de Freitas [3]
        """
        args = []
        for i in range(self.num_args):
            arg = tf.keras.layers.Dense(self.arg_depth, kernel_regularizer=tf.keras.regularizers.l2,
                                        bias_regularizer=tf.keras.regularizers.l2,
                                        name='Argument_{}'.format(str(i)))(self.h)
            args.append(arg)
        return args                                             # Shape: [bsz, arg_depth]

    def build_losses(self):
        """
        Build separate loss computations, using the logits from each of the sub-networks.
        """
        # Termination Network Loss
        termination_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.terminate, labels=self.y_term), name='Termination_Network_Loss')

        # Program Network Loss
        program_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.program_distribution, labels=self.y_prog), name='Program_Network_Loss')

        # Argument Network Losses
        arg_losses = []
        for i in range(self.num_args):
            y_arg = tf.stop_gradient(self.y_args[i])

            arg_losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.arguments[i], labels=y_arg), name='Argument{}_Network_Loss'.format(str(i))))

        return termination_loss, program_loss, arg_losses

    def build_metrics(self):
        """
        Build accuracy metrics for each of the sub-networks.
        """
        term_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.terminate, 1),
                                                      self.y_term),
                                             tf.float32), name='Termination_Accuracy')

        program_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.program_distribution, 1),
                                                         self.y_prog),
                                                tf.float32), name='Program_Accuracy')

        arg_metrics = []
        for i in range(self.num_args):
            arg_metrics.append(tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.arguments[i], 1), tf.argmax(self.y_args[i], 1)),
                        tf.float32), name='Argument{}_Accuracy'.format(str(i))))

        return term_metric, program_metric, arg_metrics
