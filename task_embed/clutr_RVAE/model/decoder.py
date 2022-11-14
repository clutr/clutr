import torch as t
import torch.nn as nn
import torch.nn.functional as F

#from utils.functional import parameters_allocation_check
#from clutr.task_embed.clutr_RVAE.utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    MINIGRID = "minigrid"
    MINIHACK = "minihack"
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params
        self.outdim = None
        if self.params.env_name == self.MINIGRID:
            self.outdim = self.params.word_vocab_size
            self.indim = self.params.latent_variable_size + self.params.word_embed_size

            self.rnn = nn.LSTM(input_size=self.indim,
                               hidden_size=self.params.decoder_rnn_size,
                               num_layers=self.params.decoder_num_layers,
                               batch_first=True)

            self.fc = nn.Linear(self.params.decoder_rnn_size, self.outdim)

        elif self.params.env_name == self.MINIHACK:
            self.outdim0 = self.params.obj_vocab_size
            self.outdim1 = self.params.word_vocab_size
            self.indim = self.params.latent_variable_size + self.params.word_embed_size + self.params.obj_embed_size

            self.rnn = nn.LSTM(input_size=self.indim,
                               hidden_size=self.params.decoder_rnn_size,
                               num_layers=self.params.decoder_num_layers,
                               batch_first=True)

            self.fc0 = nn.Linear(self.params.decoder_rnn_size, self.outdim0)
            self.fc1 = nn.Linear(self.params.decoder_rnn_size, self.outdim1)
        else:
            assert False, "Invalid Env"



    def forward(self, decoder_input, z, drop_prob, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        #assert parameters_allocation_check(self), \
        #    'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        if self.params.env_name == self.MINIGRID:
            result = self.fc(rnn_out)
            result = result.view(batch_size, seq_len, self.outdim)

        elif self.params.env_name == self.MINIHACK:
            obj = self.fc0(rnn_out).view(batch_size, seq_len, self.outdim0)
            loc = self.fc1(rnn_out).view(batch_size, seq_len, self.outdim1)
            result = t.cat((obj, loc), axis=2)

        return result, final_state
