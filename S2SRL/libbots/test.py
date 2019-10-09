import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
KEY_ATTN_SCORE = 'attention_score'

def attention(context, output):
    batch_size = output.size(0)
    hidden_size = output.size(2)
    input_size = context.size(1)
    linear_out = nn.Linear(hidden_size * 2, hidden_size)
    # (5, 256, 3)
    context_trans = context.transpose(1, 2)
    # (5, 5, 3)
    attn = torch.bmm(output, context.transpose(1, 2))
    # Returns a new tensor with the same data as the self tensor but of a different shape.
    # [4, 4] -> view(-1, 8) -> [2, 8]
    # Here .view operation is used to squeeze outputs in each batches into one batch.
    # (25, 3)
    attn_view = attn.view(-1, input_size)
    # (25, 3)
    attn1 = F.softmax(attn_view, dim=1)
    # Here .view operation is used to un-squeeze outputs into different batches.
    # (5, 5, 3)
    attn1 = attn1.view(batch_size, -1, input_size)
    attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
    # (5, 5, 256)
    mix = torch.bmm(attn, context)
    print(mix.size())
    # (5, 5, 512)
    combined = torch.cat((mix, output), dim=2)
    # (25, 512)
    combined1 = combined.view(-1, 2 * hidden_size)
    # (25, 256)
    combined1 = linear_out(combined1)
    # (25, 256)
    combined1 = torch.tanh(combined1)
    # (5, 5, 256)
    combined1 = combined1.view(batch_size, -1, hidden_size)
    output = torch.tanh(linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
    return output, attn

def forward_step(output):
    output_size = 10000
    batch_size = output.size(0)
    hidden_size = output.size(2)
    out = nn.Linear(hidden_size, output_size)
    function = F.log_softmax
    # Returns a contiguous tensor containing the same data as self tensor.
    # If self tensor is contiguous, this function returns the self tensor.
    # (5, 5, 256)
    output1 = output.contiguous()
    # (25, 256)
    output1 = output1.view(-1, hidden_size)
    # (25, 10000)
    output1 = out(output1)
    output1 = function(output1, dim=1)
    # (5, 10000, 5) TODO WHY?
    output1 = output1.view(batch_size, output_size, -1)
    predicted_softmax  = function(out(output.contiguous().view(-1, hidden_size)), dim=1).view(batch_size, output_size, -1)
    print(predicted_softmax.size())
    return predicted_softmax

if __name__ == "__main__":
    max_length = 50
    batch_size = 5
    eos_id = 1000
    ret_dict = dict()
    ret_dict[KEY_ATTN_SCORE] = list()
    context = torch.randn(5, 3, 256)
    print(context.size())
    input = torch.randn(5, 5, 256)
    print(input.size())
    # (5, 5, 256) & (5, 5, 3)
    output, attn = attention(context, input)
    # (5, 10000, 5)
    decoder_output = forward_step(output)
    decoder_outputs = []
    sequence_symbols = []
    # 1 * batch_size list in which value of each element is max_length.
    lengths = np.array([max_length] * batch_size)
    print(lengths)
    # step_output: (5, 1, 5), step_attn: (5, 1, 3)
    def decode(step, step_output, step_attn):
        decoder_outputs.append(step_output)
        ret_dict[KEY_ATTN_SCORE].append(step_attn)
        symbols1 = decoder_outputs[-1]
        symbols1 = symbols1.topk(1)
        symbols1 = symbols1[1]
        symbols = decoder_outputs[-1].topk(1)[1]
        sequence_symbols.append(symbols)
        eos_batches = symbols.data.eq(eos_id)
        if eos_batches.dim() > 0:
            eos_batches = eos_batches.cpu().view(-1).numpy()
            update_idx = ((lengths > step) & eos_batches) != 0
            lengths[update_idx] = len(sequence_symbols)
        return symbols
    for di in range(decoder_output.size(1)):
        step_output = decoder_output[:, di, :]
        # attn: (5, 5, 3)
        if attn is not None:
            step_attn = attn[:, di, :]
        else:
            step_attn = None
        decode(di, step_output, step_attn)

    # context1 = torch.randn(25, 3)
    # context1 = F.softmax(context1, dim=1)
    # context2 = context1.view(5, -1, 3)
    # print(context2.size())
    # for di in range(context2.size(1)):
    #     step_output = context2[:, di, :]
    #     print(step_output)