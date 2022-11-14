from .functional import *


class Parameters:
    def __init__(self, max_seq_len, word_vocab_size, recons_weight=79,
                 latent_variable_size=1100, vae_type="vae", enc_activation=None, env_name = None, word_embed_size=300, obj_embed_size=100,
                 activation_scalar=None):

        #self.batching = batching
        #self.loss_type = loss_type # loss_type="seq-ce"
        self.env_name = env_name
        self.enc_activation = enc_activation
        self.vae_type = vae_type
        #self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token

        self.activation_scalar = activation_scalar

        if isinstance(word_vocab_size, tuple):

            self.obj_vocab_size = int(word_vocab_size[0])
            self.obj_embed_size = obj_embed_size

            self.word_vocab_size = int(word_vocab_size[1])
            self.word_embed_size = word_embed_size

        else:
            self.word_vocab_size = int(word_vocab_size)
            self.word_embed_size = word_embed_size





        #self.char_embed_size = 15

        self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]
        self.sum_depth = fold(lambda x, y: x + y, [depth for _, depth in self.kernels], 0)

        self.encoder_rnn_size = 600
        self.encoder_num_layers = 1

        self.latent_variable_size = latent_variable_size

        self.decoder_rnn_size = 800
        self.decoder_num_layers = 2

        self.recons_weight = recons_weight