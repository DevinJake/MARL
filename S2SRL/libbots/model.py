import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from . import utils

HIDDEN_STATE_SIZE = 128
EMBEDDING_DIM = 50


class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        # # BiLSTM
        # self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
        #                        num_layers=1, batch_first=True, bidirectional=True)
        # self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
        #                                               num_layers=2, batch_first=True)

        # LSTM
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(hid_size, dict_size)
        )

    # hidden stat;
    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index+1].contiguous(), \
               encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x):
        # Example for unsqueeze:
        #             >>> x = torch.tensor([1, 2, 3, 4])
        #             >>> torch.unsqueeze(x, 0)
        #             tensor([[ 1,  2,  3,  4]])
        #             >>> torch.unsqueeze(x, 1)
        #             tensor([[ 1],
        #                     [ 2],
        #                     [ 3],
        #                     [ 4]])
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        # Self.output(out) using nn.Linear(hid_size, dict_size) to transform logits to distribution over output vocab.
        out = self.output(out)
        # squeeze: Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        res_tokens = []
        # First cur_emb is the embedding of '#BEG'.
        cur_emb = begin_emb

        # At first using the '#BEG' as first input token and hidden states from encoder as initial hidden state to predict the first output token and first decoder hidden state.
        # Then predict the output token by using last step's output token as current step's input and last step's decoder hidden state.
        for _ in range(seq_len):
            # The out_logits is the distribution over whole output vocabulary.
            # The hid is new hidden state generated from current time step.
            out_logits, hid = self.decode_one(hid, cur_emb)
            # After torch.max operation, the result is a list.
            # First element is the largest logit value in dimension-1 (each row), the second value is the index of the largest logit value.
            # >>> a = torch.randn(4, 4)
            # >>> a
            # tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            #         [ 1.1949, -1.1127, -2.2379, -0.6702],
            #         [ 1.5717, -0.9207,  0.1297, -1.8768],
            #         [-0.6172,  1.0036, -0.6060, -0.2432]])
            # >>> torch.max(a, 1)
            # (tensor([ 0.8475,  1.1949,  1.5717,  1.0036]), tensor([ 3,  0,  0,  1]))
            out_token_v = torch.max(out_logits, dim=1)[1]
            # Transform tensorflow to array and return array[0];
            out_token = out_token_v.data.cpu().numpy()[0]
            # Using current output token's embedding.
            cur_emb = self.emb(out_token_v)

            # The list of out_logits list.
            res_logits.append(out_logits)
            # The list of output tokens.
            res_tokens.append(out_token)
            # When the EOS is predicted the prediction is ended.
            if stop_at_token is not None and out_token == stop_at_token:
                break
        # torch.cat(tensors, dim=0, out=None) → Tensor
        # Concatenates the given sequence of seq tensors in the given dimension.
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        # >>> x = torch.randn(2, 3)
        # >>> x
        # tensor([[ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497]])
        # >>> torch.cat((x, x, x), 0)
        # tensor([[ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497],
        #         [ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497],
        #         [ 0.6580, -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497]])
        # >>> torch.cat((x, x, x), 1)
        # tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
        #          -1.0969, -0.4614],
        #         [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
        #          -0.5790,  0.1497]])
        # Concatenate follow rows.
        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            # Using softmax to transform logits to probabilities.
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            # np.random.choice(out_probs.shape[0], p=out_probs):
            # choose one index from out_probs.shape[0] by the probabilities associated with each entry as out_probs.
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            # Transform action to tensor and cast it to the device where begin_emb is in.
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            action_v = action_v.cuda()
            # Get the embedding of the sampled output token.
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None and action == stop_at_token:
                break
        return torch.cat(res_logits), res_actions


def pack_batch_no_out(batch, embeddings, device="cpu"):
    # Assert statements are a convenient way to insert debugging assertions into a program.
    # To guarantee that the batch is a list.
    assert isinstance(batch, list)
    # The format of batch is a list of tuple: ((tuple),[[list of token ID list]])
    # A lambda function is a small anonymous function, the example is as following.
    # x = lambda a, b: a * b
    # print(x(5, 6))
    # Sort descending (CuDNN requirements) batch中第一个元素为最长的句子；
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    # input_idx：一个batch的输入句子的tokens对应的ID矩阵；Each row is corresponding to one input sentence.
    # output_idx：一个batch的输出句子的tokens对应的ID矩阵；Each row is corresponding to a list of several output sentences.
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    # map() function returns a list of the results after applying the given function to each item of a given iterable (list, tuple etc.)
    # For example:
    # numbers = (1, 2, 3, 4)
    # result = map(lambda x: x + x, numbers)
    # print(list(result))
    # Output: {2, 4, 6, 8}
    # 建立长度词典，为batch中每一个元素的长度；
    lens = list(map(len, input_idx))
    # 以最长的句子来建立batch*最长句子长度的全0矩阵；
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    # 将batch中每个句子的tokens对应的ID向量填入全0矩阵完成padding；
    # idx：index，x：token ID 组成的向量；
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    # 将padding后的矩阵转换为tensor matrix；
    input_v = torch.tensor(input_mat).to(device)
    input_v = input_v.cuda()
    # 封装成PackedSequence类型的对象；
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    input_seq = input_seq.cuda()
    # lookup embeddings；embeddings为模型已经建立的词向量矩阵；
    r = embeddings(input_seq.data)
    r = r.cuda()
    # 加入了词嵌入的input_seq；
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
    emb_input_seq = emb_input_seq.cuda()
    return emb_input_seq, input_idx, output_idx


def pack_input(input_data, embeddings, device="cpu"):
    input_v = torch.LongTensor([input_data]).to(device)
    input_v = input_v.cuda()
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)


def pack_batch(batch, embeddings, device="cpu"):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        output_seq_list.append(pack_input(out[:-1], embeddings, device))
    return emb_input_seq, output_seq_list, input_idx, output_idx


def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq, ref_seq)
