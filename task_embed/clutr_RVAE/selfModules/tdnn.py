import torch as t
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, params):
        super(TDNN, self).__init__()

        self.params = params

        self.kernels = [Parameter(t.Tensor(out_dim, self.params.char_embed_size, kW).uniform_(-1, 1))
                        for kW, out_dim in params.kernels]
        self._add_to_parameters(self.kernels, 'TDNN_kernel')

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]

        :return: tensor with shape [batch_size, max_seq_len, depth_sum]

        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert input_size_len == 4, \
            'Wrong input rang, must be equal to 4, but {} found'.format(input_size_len)

        [batch_size, seq_len, _, embed_size] = input_size

        assert embed_size == self.params.char_embed_size, \
            'Wrong embedding size, must be equal to {}, but {} found'.format(self.params.char_embed_size, embed_size)

        # leaps with shape
        x = x.view(-1, self.params.max_word_len, self.params.char_embed_size).transpose(1, 2).contiguous()

        xs = [F.tanh(F.conv1d(x, kernel)) for kernel in self.kernels]
        xs = [x.max(2)[0].squeeze(-2) for x in xs]

        x = t.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)
