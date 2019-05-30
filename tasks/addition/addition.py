"""
addition.py

Core task-specific model definition file. Sets up encoder model, program embeddings, argument
handling.
"""
from tasks.addition.env.config import CONFIG
import tensorflow as tf


class AdditionCore():
    def __init__(self, hidden_dim=100, state_dim=128, batch_size=1):
        """
        Instantiate an Addition Core object, with the necessary hyperparameters.
        """
        self.hidden_dim, self.state_dim, self.bsz = hidden_dim, state_dim, batch_size
        self.env_dim = CONFIG["ENVIRONMENT_ROW"] * CONFIG["ENVIRONMENT_DEPTH"]  # 4 * 10 = 40
        self.arg_dim = CONFIG["ARGUMENT_NUM"] * CONFIG["ARGUMENT_DEPTH"]        # 3 * 10 = 30
        self.program_dim = CONFIG["PROGRAM_EMBEDDING_SIZE"]

        # Setup Environment Input Layer
        self.env_in = tf.placeholder(tf.float32, shape=[self.bsz, self.env_dim], name="Env_Input")

        # Setup Argument Input Layer
        self.arg_in = tf.placeholder(tf.float32, shape=[self.bsz, self.arg_dim], name="Arg_Input")

        # Setup Program ID Input Layer
        self.prg_in = tf.placeholder(tf.int32, shape=[self.bsz, 1], name='Program_ID')

        # Build Environment Encoder Network (f_enc)
        self.state_encoding = self.build_encoder()

        # Build Program Matrices
        self.program_key = tf.get_variable(name='Program_Keys',
                                           shape=[CONFIG["PROGRAM_NUM"], CONFIG["PROGRAM_KEY_SIZE"]],
                                           initializer=tf.truncated_normal_initializer)
        self.program_embedding = self.build_program_store()

    def build_encoder(self):
        """
        Build the Encoder Network (f_enc) taking the environment state (env_in) and the program
        arguments (arg_in), feeding through a Multilayer Perceptron, to generate the state encoding
        (s_t).

        Reed, de Freitas only specify that the f_enc is a Multilayer Perceptron => As such we use
        two ELU Layers, up-sampling to a state vector with dimension 128.

        Reference: Reed, de Freitas [9]
        """
        merge = tf.concat([self.env_in, self.arg_in], 1)
        elu = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.elu,
                                    kernel_initializer=tf.truncated_normal_initializer)(merge)
        elu = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.elu,
                                    kernel_initializer=tf.truncated_normal_initializer)(elu)
        out = tf.keras.layers.Dense(self.state_dim,
                                    kernel_initializer=tf.truncated_normal_initializer)(elu)
        return out

    def build_program_store(self):
        """
        Build the Program Embedding (M_prog) that takes in a specific Program ID (prg_in), and
        returns the respective Program Embedding.

        Reference: Reed, de Freitas [4]
        """
        embedding_matrix = tf.get_variable(name="Embedding_Matrix",
                                           shape=[CONFIG["PROGRAM_NUM"], CONFIG["PROGRAM_EMBEDDING_SIZE"]],
                                           initializer=tf.truncated_normal_initializer)
        embedding = tf.nn.embedding_lookup(embedding_matrix, self.prg_in, name='Program_Embedding')
        return embedding
