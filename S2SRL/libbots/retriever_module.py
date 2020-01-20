import torch.nn as nn
import torch
import torch.nn.functional as F

class RetrieverModel(nn.Module):
    def __init__(self, emb_size, dict_size, EMBED_FLAG=False, hid1_size = 300, hid2_size = 200, output_size = 128, device='cpu'):
        # Call __init__ function of PhraseModel's parent class (nn.Module).
        super(RetrieverModel, self).__init__()
        # With :attr:`padding_idx` set, the embedding vector at
        #         :attr:`padding_idx` is initialized to all zeros. However, notice that this
        #         vector can be modified afterwards, e.g., using a customized
        #         initialization method, and thus changing the vector used to pad the
        #         output. The gradient for this vector from :class:`~torch.nn.Embedding`
        #         is always zero.
        self.document_emb = nn.Embedding(num_embeddings=dict_size+1, embedding_dim=emb_size, padding_idx=dict_size)
        if not EMBED_FLAG:
            for p in self.parameters():
                p.requires_grad = False
        self.hid_layer1 = nn.Linear(emb_size, hid1_size)
        self.hid_layer2 = nn.Linear(hid1_size, hid2_size)
        self.output_layer = nn.Linear(hid2_size, output_size)
        self.device = device

    def forward(self, query_tensor, range):
        documents = self.pack_input(range)
        query_tensor = self.hid_layer1(query_tensor)
        query_tensor = self.hid_layer2(query_tensor)
        query_tensor = self.output_layer(query_tensor)
        documents = self.hid_layer1(documents)
        documents = self.hid_layer2(documents)
        documents = self.output_layer(documents)
        cosine_output = torch.cosine_similarity(query_tensor, documents, dim=1)
        logsoftmax_output = F.log_softmax(cosine_output, dim=0)
        softmax_output = F.softmax(cosine_output, dim=0)
        return logsoftmax_output, softmax_output, cosine_output

    def get_retriever_net_parameter(self):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        params = self.named_parameters()
        param_dict = dict()
        for name, param in params:
            param_dict[name] = param.to(self.device).clone().detach()
        return param_dict

    def pack_input(self, indices):
        dict_size = self.document_emb.weight.shape[0]-1
        if not isinstance(indices, tuple):
            index = indices
            if index >= dict_size or index < 0:
                index = dict_size
            input_v = torch.LongTensor([index]).to(self.device)
            input_v = input_v.cuda()
            r = self.document_emb(input_v)
            return r
        else:
            list = [dict_size if (i >= dict_size or i < 0) else i for i in range(indices[0], indices[1])]
            input_v = torch.LongTensor(list).to(self.device)
            input_v = input_v.cuda()
            r = self.document_emb(input_v)
            return r

    @classmethod
    def calculate_rank(self, vector):
        rank = 1
        order_list = {}
        if isinstance(vector, list):
            for value in sorted(vector, reverse=True):
                if value not in order_list:
                    order_list[value] = rank
                rank += 1
            order = [order_list[i] for i in vector]
            return order
        else:
            return []