import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::
         # >>> attention = seq2seq.models.Attention(256)
         # >>> context = Variable(torch.randn(5, 3, 256))
         # >>> output = Variable(torch.randn(5, 5, 256))
         # >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        unpack_output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        batch_size = unpack_output.size(0)
        hidden_size = unpack_output.size(2)
        context_trans = context.view(1, -1, context.size(1))
        input_size = context_trans.size(1)
        # for idx1, temp1 in enumerate(unpack_output[0]):
        #     for idx, temp in enumerate(context_trans[0]):
        #         print('o'+str(idx1)+',c'+str(idx)+': '+str(torch.dot(temp1, temp)))
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(unpack_output, context_trans.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context_trans)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, unpack_output), dim=2)
        # output -> (batch, out_len, dim)
        output_result = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(-1, hidden_size)
        # Transform result into PackedSequence format.
        packed_output_result = rnn_utils.PackedSequence(output_result, output.batch_sizes.detach())
        return packed_output_result, attn
